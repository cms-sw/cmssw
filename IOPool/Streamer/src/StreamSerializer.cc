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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "zlib.h"
#include "lzma.h"
#include "zstd.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace edm {

  /**
   * Creates a translator instance for the specified product registry.
   */
  StreamSerializer::StreamSerializer(SelectedProducts const *selections)
      : selections_(selections), tc_(getTClass(typeid(SendEvent))) {}

  /**
   * Serializes the product registry (that was specified to the constructor)
   * into the specified InitMessage.
   */
  int StreamSerializer::serializeRegistry(SerializeDataBuffer &data_buffer,
                                          const BranchIDLists &branchIDLists,
                                          ThinnedAssociationsHelper const &thinnedAssociationsHelper) {
    SendJobHeader::ParameterSetMap psetMap;
    pset::Registry::instance()->fillMap(psetMap);
    return serializeRegistry(data_buffer, branchIDLists, thinnedAssociationsHelper, psetMap);
  }

  int StreamSerializer::serializeRegistry(SerializeDataBuffer &data_buffer,
                                          const BranchIDLists &branchIDLists,
                                          ThinnedAssociationsHelper const &thinnedAssociationsHelper,
                                          SendJobHeader::ParameterSetMap const &psetMap) {
    FDEBUG(6) << "StreamSerializer::serializeRegistry" << std::endl;
    SendJobHeader sd;

    FDEBUG(9) << "Product List: " << std::endl;

    for (auto const &selection : *selections_) {
      sd.push_back(*selection.first);
      FDEBUG(9) << "StreamOutput got product = " << selection.first->className() << std::endl;
    }
    Service<ConstProductRegistry> reg;
    sd.setBranchIDLists(branchIDLists);
    sd.setThinnedAssociationsHelper(thinnedAssociationsHelper);
    sd.setParameterSetMap(psetMap);

    data_buffer.rootbuf_.Reset();

    RootDebug tracer(10, 10);

    TClass *tc = getTClass(typeid(SendJobHeader));
    int bres = data_buffer.rootbuf_.WriteObjectAny((char *)&sd, tc);

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
    data_buffer.ptr_ = (unsigned char *)data_buffer.rootbuf_.Buffer();
    // calculate the adler32 checksum and fill it into the struct
    data_buffer.adler32_chksum_ = cms::Adler32((char *)data_buffer.bufferPointer(), data_buffer.curr_space_used_);
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
  int StreamSerializer::serializeEvent(SerializeDataBuffer &data_buffer,
                                       EventForOutput const &event,
                                       ParameterSetID const &selectorConfig,
                                       StreamerCompressionAlgo compressionAlgo,
                                       int compression_level,
                                       unsigned int reserveSize) const {
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

    for (auto const &selection : *selections_) {
      BranchDescription const &desc = *selection.first;
      BasicHandle result = event.getByToken(selection.second, desc.unwrappedTypeID());
      if (!result.isValid()) {
        // No product with this ID was put in the event.
        // Create and write the provenance.
        se.products().push_back(StreamedProduct(desc));
      } else {
        if (result.provenance()->productProvenance()) {
          Parentage const *parentage =
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
    data_buffer.ptr_ = (unsigned char *)data_buffer.rootbuf_.Buffer();

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
    unsigned int dest_size = 0;
    switch (compressionAlgo) {
      case ZLIB:
        dest_size = compressBuffer((unsigned char *)data_buffer.rootbuf_.Buffer(),
                                   data_buffer.curr_event_size_,
                                   data_buffer.comp_buf_,
                                   compression_level,
                                   reserveSize);
        break;
      case LZMA:
        dest_size = compressBufferLZMA((unsigned char *)data_buffer.rootbuf_.Buffer(),
                                       data_buffer.curr_event_size_,
                                       data_buffer.comp_buf_,
                                       compression_level,
                                       reserveSize);
        break;
      case ZSTD:
        dest_size = compressBufferZSTD((unsigned char *)data_buffer.rootbuf_.Buffer(),
                                       data_buffer.curr_event_size_,
                                       data_buffer.comp_buf_,
                                       compression_level,
                                       reserveSize);
        break;
      default:
        dest_size = data_buffer.rootbuf_.Length();
        if (data_buffer.comp_buf_.size() < dest_size + reserveSize)
          data_buffer.comp_buf_.resize(dest_size + reserveSize);
        std::copy((char *)data_buffer.rootbuf_.Buffer(),
                  (char *)data_buffer.rootbuf_.Buffer() + dest_size,
                  (char *)(&data_buffer.comp_buf_[SerializeDataBuffer::reserve_size]));
        break;
    };

    data_buffer.ptr_ = &data_buffer.comp_buf_[reserveSize];  // reset to point at compressed area
    data_buffer.curr_space_used_ = dest_size;

    // calculate the adler32 checksum and fill it into the struct
    data_buffer.adler32_chksum_ = cms::Adler32((char *)data_buffer.bufferPointer(), data_buffer.curr_space_used_);
    //std::cout << "Adler32 checksum of event = " << data_buffer.adler32_chksum_ << std::endl;

    return data_buffer.curr_space_used_;
  }

  /**
   * Compresses the data in the specified input buffer into the
   * specified output buffer.  Returns the size of the compressed data
   * or zero if compression failed.
   */
  unsigned int StreamSerializer::compressBuffer(unsigned char *inputBuffer,
                                                unsigned int inputSize,
                                                std::vector<unsigned char> &outputBuffer,
                                                int compressionLevel,
                                                unsigned int reserveSize) {
    unsigned int resultSize = 0;

    // what are these magic numbers? (jbk) -> LSB 3.0 buffer size reccommendation
    unsigned long dest_size = (unsigned long)(double(inputSize) * 1.002 + 1.0) + 12;
    //this can has some overhead in memory usage (capacity > size) due to the way std::vector allocator works
    if (outputBuffer.size() < dest_size + reserveSize)
      outputBuffer.resize(dest_size + reserveSize);

    // compression 1-9, 6 is zlib default, 0 none
    int ret = compress2(&outputBuffer[reserveSize], &dest_size, inputBuffer, inputSize, compressionLevel);

    // check status
    if (ret == Z_OK) {
      // return the correct length
      resultSize = dest_size;

      FDEBUG(1) << " original size = " << inputSize << " final size = " << dest_size
                << " ratio = " << double(dest_size) / double(inputSize) << std::endl;
    } else {
      throw cms::Exception("StreamSerializer", "compressBuffer")
          << "Compression Return value: " << ret << " Okay = " << Z_OK << std::endl;
    }

    return resultSize;
  }

  //this is based on ROOT R__zipLZMA
  unsigned int StreamSerializer::compressBufferLZMA(unsigned char *inputBuffer,
                                                    unsigned int inputSize,
                                                    std::vector<unsigned char> &outputBuffer,
                                                    int compressionLevel,
                                                    unsigned int reserveSize,
                                                    bool addHeader) {
    // what are these magic numbers? (jbk)
    unsigned int hdr_size = addHeader ? 4 : 0;
    unsigned long dest_size = (unsigned long)(double(inputSize) * 1.01 + 1.0) + 12;
    if (outputBuffer.size() < dest_size + reserveSize)
      outputBuffer.resize(dest_size + reserveSize);

    // compression 1-9
    uint32_t dict_size_est = inputSize / 4;
    lzma_stream stream = LZMA_STREAM_INIT;
    lzma_options_lzma opt_lzma2;
    lzma_filter filters[] = {
        {.id = LZMA_FILTER_LZMA2, .options = &opt_lzma2},
        {.id = LZMA_VLI_UNKNOWN, .options = nullptr},
    };
    lzma_ret returnStatus;

    unsigned char *tgt = &outputBuffer[reserveSize];

    //if (*srcsize > 0xffffff || *srcsize < 0) { //16 MB limit ?
    //   return;
    //}

    if (compressionLevel > 9)
      compressionLevel = 9;

    lzma_bool presetStatus = lzma_lzma_preset(&opt_lzma2, compressionLevel);
    if (presetStatus) {
      throw cms::Exception("StreamSerializer", "compressBufferLZMA") << "LZMA preset return status: " << presetStatus;
    }

    if (LZMA_DICT_SIZE_MIN > dict_size_est) {
      dict_size_est = LZMA_DICT_SIZE_MIN;
    }
    if (opt_lzma2.dict_size > dict_size_est) {
      /* reduce the dictionary size if larger than 1/4 the input size, preset
         dictionaries size can be expensively large
       */
      opt_lzma2.dict_size = dict_size_est;
    }

    returnStatus =
        lzma_stream_encoder(&stream,
                            filters,
                            LZMA_CHECK_NONE);  //CRC32 and CRC64 are available, but we already calculate adler32
    if (returnStatus != LZMA_OK) {
      throw cms::Exception("StreamSerializer", "compressBufferLZMA")
          << "LZMA compression encoder return value: " << returnStatus;
    }

    stream.next_in = (const uint8_t *)inputBuffer;
    stream.avail_in = (size_t)(inputSize);

    stream.next_out = (uint8_t *)(&tgt[hdr_size]);
    stream.avail_out = (size_t)(dest_size - hdr_size);

    returnStatus = lzma_code(&stream, LZMA_FINISH);

    if (returnStatus != LZMA_STREAM_END) {
      lzma_end(&stream);
      throw cms::Exception("StreamSerializer", "compressBufferLZMA")
          << "LZMA compression return value: " << returnStatus;
    }
    lzma_end(&stream);

    //Add compression-specific header at the buffer start. This will be used to detect LZMA(2) format after streamer header
    if (addHeader) {
      tgt[0] = 'X'; /* Signature of LZMA from XZ Utils */
      tgt[1] = 'Z';
      tgt[2] = 0;
      tgt[3] = 0;  //let's put offset to 4, not 3
    }

    FDEBUG(1) << " LZMA original size = " << inputSize << " final size = " << stream.total_out
              << " ratio = " << double(stream.total_out) / double(inputSize) << std::endl;

    return stream.total_out + hdr_size;
  }

  unsigned int StreamSerializer::compressBufferZSTD(unsigned char *inputBuffer,
                                                    unsigned int inputSize,
                                                    std::vector<unsigned char> &outputBuffer,
                                                    int compressionLevel,
                                                    unsigned int reserveSize,
                                                    bool addHeader) {
    unsigned int hdr_size = addHeader ? 4 : 0;
    unsigned int resultSize = 0;

    // what are these magic numbers? (jbk) -> LSB 3.0 buffer size reccommendation
    size_t worst_size = ZSTD_compressBound(inputSize);
    //this can has some overhead in memory usage (capacity > size) due to the way std::vector allocator works
    if (outputBuffer.size() < worst_size + reserveSize + hdr_size)
      outputBuffer.resize(worst_size + reserveSize + hdr_size);

    //Add compression-specific header at the buffer start. This will be used to detect ZSTD format after streamer header
    unsigned char *tgt = &outputBuffer[reserveSize];
    if (addHeader) {
      tgt[0] = 'Z'; /* Pre */
      tgt[1] = 'S';
      tgt[2] = 0;
      tgt[3] = 0;
    }

    // compression 1-20
    size_t dest_size = ZSTD_compress(
        (void *)&outputBuffer[reserveSize + hdr_size], worst_size, (void *)inputBuffer, inputSize, compressionLevel);

    // check status
    if (!ZSTD_isError(dest_size)) {
      // return the correct length
      resultSize = (unsigned int)dest_size + hdr_size;

      FDEBUG(1) << " original size = " << inputSize << " final size = " << dest_size
                << " ratio = " << double(dest_size) / double(inputSize) << std::endl;
    } else {
      throw cms::Exception("StreamSerializer", "compressBuffer")
          << "Compression (ZSTD) Error: " << ZSTD_getErrorName(dest_size);
    }

    return resultSize;
  }

}  // namespace edm
