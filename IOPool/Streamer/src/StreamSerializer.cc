/**
 * StreamSerializer.cc
 *
 * Utility class for serializing framework objects (e.g. ProductRegistry and
 * Event) into streamer message objects.
 */
#include "IOPool/Streamer/interface/StreamSerializer.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/SelectedProducts.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "zlib.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace edm {

  /**
   * Creates a translator instance for the specified product registry.
   */
  StreamSerializer::StreamSerializer(SelectedProducts const* selections)
      : selections_(selections), tc_(getTClass(typeid(SendEvent))) {}

  /**
   * Serializes the product registry (that was specified to the constructor)
   * into the specified InitMessage.
   */

  int StreamSerializer::serializeRegistry(SerializeDataBuffer& data_buffer,
                                          const BranchIDLists& branchIDLists,
                                          ThinnedAssociationsHelper const& thinnedAssociationsHelper) {
    FDEBUG(6) << "StreamSerializer::serializeRegistry" << std::endl;
    SendJobHeader sd;

    FDEBUG(9) << "Product List: " << std::endl;

    for (auto const& selection : *selections_) {
      sd.push_back(*selection.first);
      FDEBUG(9) << "StreamOutput got product = " << selection.first->className() << std::endl;
    }
    Service<ConstProductRegistry> reg;
    sd.setBranchIDLists(branchIDLists);
    sd.setThinnedAssociationsHelper(thinnedAssociationsHelper);
    SendJobHeader::ParameterSetMap psetMap;

    pset::Registry::instance()->fillMap(psetMap);
    sd.setParameterSetMap(psetMap);

    data_buffer.rootbuf_.Reset();

    RootDebug tracer(10, 10);

    TClass* tc = getTClass(typeid(SendJobHeader));
    int bres = data_buffer.rootbuf_.WriteObjectAny((char*)&sd, tc);

    switch (bres) {
      case 0:  // failure
      {
        throw cms::Exception("StreamTranslation", "Registry serialization failed")
            << "StreamSerializer failed to serialize registry\n";
        break;
      }
      case 1:  // succcess
        break;
      case 2:  // truncated result
      {
        throw cms::Exception("StreamTranslation", "Registry serialization truncated")
            << "StreamSerializer module attempted to serialize\n"
            << "a registry that is to big for the allocated buffers\n";
        break;
      }
      default:  // unknown
      {
        throw cms::Exception("StreamTranslation", "Registry serialization failed")
            << "StreamSerializer module got an unknown error code\n"
            << " while attempting to serialize registry\n";
        break;
      }
    }

    data_buffer.curr_event_size_ = data_buffer.rootbuf_.Length();
    data_buffer.curr_space_used_ = data_buffer.curr_event_size_;
    data_buffer.ptr_ = (unsigned char*)data_buffer.rootbuf_.Buffer();
    // calculate the adler32 checksum and fill it into the struct
    data_buffer.adler32_chksum_ = cms::Adler32((char*)data_buffer.bufferPointer(), data_buffer.curr_space_used_);
    //std::cout << "Adler32 checksum of init message = " << data_buffer.adler32_chksum_ << std::endl;
    return data_buffer.curr_space_used_;
  }

  /**
   * Serializes the specified event into the specified event message.


   make a char* as a data member, tell ROOT to not adapt it, but still use it.
   initialize it to 1M, let ROOT resize if it wants, then delete it in the
   dtor.

   change the call to not take an eventMessage, add a member function to
   return the address of the place that ROOT wrote the serialized data.

   return the length of the serialized object and the actual length if
   compression has been done (may want to cache these lengths in this
   object instead.

   the caller will need to copy the data from this object to its final
   destination in the EventMsgBuilder.


   */
  int StreamSerializer::serializeEvent(EventForOutput const& event,
                                       ParameterSetID const& selectorConfig,
                                       bool use_compression,
                                       int compression_level,
                                       SerializeDataBuffer& data_buffer) const {
    EventSelectionIDVector selectionIDs = event.eventSelectionIDs();
    selectionIDs.push_back(selectorConfig);
    SendEvent se(event.eventAuxiliary(), event.processHistory(), selectionIDs, event.branchListIndexes());

    // Loop over EDProducts, fill the provenance, and write.

    // Historical note. I fixed two bugs in the code below in
    // March 2017. One would have caused any Parentage written
    // using the Streamer output module to be total nonsense
    // prior to the fix. The other would have caused seg faults
    // when the Parentage was dropped in an earlier process.

    // FIX ME. The code below stores the direct parentage of
    // kept products, but it does not save the parentage of
    // dropped objects that are ancestors of kept products like
    // the PoolOutputModule. That information is currently
    // lost when the streamer output module is used.

    for (auto const& selection : *selections_) {
      BranchDescription const& desc = *selection.first;
      BasicHandle result = event.getByToken(selection.second, desc.unwrappedTypeID());
      if (!result.isValid()) {
        // No product with this ID was put in the event.
        // Create and write the provenance.
        se.products().push_back(StreamedProduct(desc));
      } else {
        if (result.provenance()->productProvenance()) {
          Parentage const* parentage =
              ParentageRegistry::instance()->getMapped(result.provenance()->productProvenance()->parentageID());
          assert(parentage);
          se.products().push_back(
              StreamedProduct(result.wrapper(), desc, result.wrapper() != nullptr, &parentage->parents()));
        } else {
          se.products().push_back(StreamedProduct(result.wrapper(), desc, result.wrapper() != nullptr, nullptr));
        }
      }
    }

    data_buffer.rootbuf_.Reset();
    RootDebug tracer(10, 10);

    //TClass* tc = getTClass(typeid(SendEvent));
    int bres = data_buffer.rootbuf_.WriteObjectAny(&se, tc_);
    switch (bres) {
      case 0:  // failure
      {
        throw cms::Exception("StreamTranslation", "Event serialization failed")
            << "StreamSerializer failed to serialize event: " << event.id();
        break;
      }
      case 1:  // succcess
        break;
      case 2:  // truncated result
      {
        throw cms::Exception("StreamTranslation", "Event serialization truncated")
            << "StreamSerializer module attempted to serialize an event\n"
            << "that is to big for the allocated buffers: " << event.id();
        break;
      }
      default:  // unknown
      {
        throw cms::Exception("StreamTranslation", "Event serialization failed")
            << "StreamSerializer module got an unknown error code\n"
            << " while attempting to serialize event: " << event.id();
        break;
      }
    }

    data_buffer.curr_event_size_ = data_buffer.rootbuf_.Length();
    data_buffer.curr_space_used_ = data_buffer.curr_event_size_;
    data_buffer.ptr_ = (unsigned char*)data_buffer.rootbuf_.Buffer();
#if 0
   if(data_buffer.ptr_ != data_.ptr_) {
        std::cerr << "ROOT reset the buffer!!!!\n";
        data_.ptr_ = data_buffer.ptr_; // ROOT may have reset our data pointer!!!!
        }
#endif
    // std::copy(rootbuf_.Buffer(),rootbuf_.Buffer()+rootbuf_.Length(),
    // eventMessage.eventAddr());
    // eventMessage.setEventLength(rootbuf.Length());

    // compress before return if we need to
    // should test if compressed already - should never be?
    //   as double compression can have problems
    if (use_compression) {
      unsigned int dest_size =
          compressBuffer(data_buffer.ptr_, data_buffer.curr_event_size_, data_buffer.comp_buf_, compression_level);
      if (dest_size != 0) {
        data_buffer.ptr_ = &data_buffer.comp_buf_[0];  // reset to point at compressed area
        data_buffer.curr_space_used_ = dest_size;
      }
    }
    // calculate the adler32 checksum and fill it into the struct
    data_buffer.adler32_chksum_ = cms::Adler32((char*)data_buffer.bufferPointer(), data_buffer.curr_space_used_);
    //std::cout << "Adler32 checksum of event = " << data_buffer.adler32_chksum_ << std::endl;

    return data_buffer.curr_space_used_;
  }

  /**
   * Compresses the data in the specified input buffer into the
   * specified output buffer.  Returns the size of the compressed data
   * or zero if compression failed.
   */
  unsigned int StreamSerializer::compressBuffer(unsigned char* inputBuffer,
                                                unsigned int inputSize,
                                                std::vector<unsigned char>& outputBuffer,
                                                int compressionLevel) {
    unsigned int resultSize = 0;

    // what are these magic numbers? (jbk)
    unsigned long dest_size = (unsigned long)(double(inputSize) * 1.002 + 1.0) + 12;
    if (outputBuffer.size() < dest_size)
      outputBuffer.resize(dest_size);

    // compression 1-9, 6 is zlib default, 0 none
    int ret = compress2(&outputBuffer[0], &dest_size, inputBuffer, inputSize, compressionLevel);

    // check status
    if (ret == Z_OK) {
      // return the correct length
      resultSize = dest_size;

      FDEBUG(1) << " original size = " << inputSize << " final size = " << dest_size
                << " ratio = " << double(dest_size) / double(inputSize) << std::endl;
    } else {
      // compression failed, return a size of zero
      FDEBUG(9) << "Compression Return value: " << ret << " Okay = " << Z_OK << std::endl;
      // do we throw an exception here?
      std::cerr << "Compression Return value: " << ret << " Okay = " << Z_OK << std::endl;
    }

    return resultSize;
  }
}  // namespace edm
