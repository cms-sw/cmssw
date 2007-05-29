/**
 * StreamSerializer.cc
 *
 * Utility class for serializing framework objects (e.g. ProductRegistry and
 * EventPrincipal) into streamer message objects.
 */

#include "IOPool/Streamer/interface/StreamSerializer.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "DataFormats/Common/interface/BasicHandle.h"

#include "zlib.h"
#include <cstdlib>

using namespace std;

namespace edm
{

  StreamSerializer::Arr::Arr(int sz):ptr_((char*)malloc(sz)) { }
  StreamSerializer::Arr::~Arr() { free(ptr_); }

  const int init_size = 1024*1024;

  /**
   * Creates a translator instance for the specified product registry.
   */
  StreamSerializer::StreamSerializer(OutputModule::Selections const* selections):
    selections_(selections),
    //data_(init_size),
    comp_buf_(init_size),
    curr_event_size_(),
    curr_space_used_(),
    rootbuf_(TBuffer::kWrite,init_size), // ,data_.ptr_,kFALSE),
    ptr_((unsigned char*)rootbuf_.Buffer()),
    tc_(getTClass(typeid(SendEvent)))
  { }

  /**
   * Serializes the product registry (that was specified to the constructor)
   * into the specified InitMessage.
   */
  int StreamSerializer::serializeRegistry(InitMsgBuilder& initMessage)
  {
    FDEBUG(6) << "StreamSerializer::serializeRegistry" << endl;
    TClass* prog_reg = getTClass(typeid(SendJobHeader));
    SendJobHeader sd;

    OutputModule::Selections::const_iterator i(selections_->begin()),e(selections_->end());

    FDEBUG(9) << "Product List: " << endl;

    for(; i != e; ++i)  
      {
        sd.descs_.push_back(**i);
        FDEBUG(9) << "StreamOutput got product = " << (*i)->className()
                  << endl;
      }

    TBuffer rootbuf(TBuffer::kWrite,initMessage.bufferSize(),
                    initMessage.dataAddress(),kFALSE);

    RootDebug tracer(10,10);

    int bres = rootbuf.WriteObjectAny((char*)&sd,prog_reg);

    switch(bres)
      {
      case 0: // failure
        {
          throw cms::Exception("StreamTranslation","Registry serialization failed")
            << "StreamSerializer failed to serialize registry\n";
          break;
        }
      case 1: // succcess
        break;
      case 2: // truncated result
        {
          throw cms::Exception("StreamTranslation","Registry serialization truncated")
            << "StreamSerializer module attempted to serialize\n"
            << "a registry that is to big for the allocated buffers\n";
          break;
        }
      default: // unknown
        {
          throw cms::Exception("StreamTranslation","Registry serialization failed")
            << "StreamSerializer module got an unknown error code\n"
            << " while attempting to serialize registry\n";
          break;
        }
      }

    initMessage.setDescLength(rootbuf.Length());
    return rootbuf.Length();
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
  int StreamSerializer::serializeEvent(EventPrincipal const& eventPrincipal,
                                       bool use_compression, 
				       int compression_level)
  {
    SendEvent se(eventPrincipal.id(),eventPrincipal.time());

    OutputModule::Selections::const_iterator i(selections_->begin()),ie(selections_->end());
    // Loop over EDProducts, fill the provenance, and write.

    for(; i != ie; ++i) {
      BranchDescription const& desc = **i;
      ProductID const& id = desc.productID();

      if (id == ProductID()) {
        throw Exception(errors::ProductNotFound,"InvalidID")
          << "StreamSerializer::serializeEvent: invalid ProductID supplied in productRegistry\n";
      }
      BasicHandle const bh = eventPrincipal.getForOutput(id, true);
      if (bh.provenance() == 0) {
        std::string const& name = desc.className();
        std::string const className = wrappedClassName(name);
        TClass *cp = gROOT->GetClass(className.c_str());
        if (cp == 0) {
          throw Exception(errors::ProductNotFound,"NoMatch")
            << "TypeID::className: No dictionary for class " << className << '\n'
            << "Add an entry for this class\n"
            << "to the appropriate 'classes_def.xml' and 'classes.h' files." << '\n';
        }


        EDProduct *p = static_cast<EDProduct *>(cp->New());
        se.prods_.push_back(ProdPair(p, bh.provenance()));
      } else {
        se.prods_.push_back(ProdPair(bh.wrapper(), bh.provenance()));
      }
     }

    //TBuffer rootbuf(TBuffer::kWrite,eventMessage.bufferSize(),
    //                eventMessage.eventAddr(),kFALSE);

    rootbuf_.Reset();
    RootDebug tracer(10,10);

    //TClass* tc = getTClass(typeid(SendEvent));
    int bres = rootbuf_.WriteObjectAny(&se,tc_);
   switch(bres)
      {
      case 0: // failure
        {
          throw cms::Exception("StreamTranslation","Event serialization failed")
            << "StreamSerializer failed to serialize event: "
            << eventPrincipal.id();
          break;
        }
      case 1: // succcess
        break;
      case 2: // truncated result
        {
          throw cms::Exception("StreamTranslation","Event serialization truncated")
            << "StreamSerializer module attempted to serialize an event\n"
            << "that is to big for the allocated buffers: "
            << eventPrincipal.id();
          break;
        }
    default: // unknown
        {
          throw cms::Exception("StreamTranslation","Event serialization failed")
            << "StreamSerializer module got an unknown error code\n"
            << " while attempting to serialize event: "
            << eventPrincipal.id();
          break;
        }
      }
   
   curr_event_size_ = rootbuf_.Length();
   curr_space_used_ = curr_event_size_;
   ptr_ = (unsigned char*)rootbuf_.Buffer();
#if 0
   if(ptr_ != data_.ptr_)
	{
	cerr << "ROOT reset the buffer!!!!\n";
	data_.ptr_ = ptr_; // ROOT may have reset our data pointer!!!!
	}
#endif
   // copy(rootbuf_.Buffer(),rootbuf_.Buffer()+rootbuf_.Length(),
   //	eventMessage.eventAddr());
   // eventMessage.setEventLength(rootbuf.Length()); 

    // compress before return if we need to
    // should test if compressed already - should never be?
    //   as double compression can have problems
    if(use_compression)
    {
      unsigned int dest_size =
        compressBuffer(ptr_, curr_event_size_, comp_buf_, compression_level);
      if(dest_size != 0)
      {
	ptr_ = &comp_buf_[0]; // reset to point at compressed area
        curr_space_used_ = dest_size;
      }
    }

    return curr_space_used_;
  }

  /**
   * Compresses the data in the specified input buffer into the
   * specified output buffer.  Returns the size of the compressed data
   * or zero if compression failed.
   */
  unsigned int
  StreamSerializer::compressBuffer(unsigned char *inputBuffer,
				   unsigned int inputSize,
				   std::vector<unsigned char> &outputBuffer,
				   int compressionLevel)
  {
    unsigned int resultSize = 0;

    // what are these magic numbers? (jbk)
    unsigned long dest_size = (unsigned long)(double(inputSize)*
					      1.002 + 1.0) + 12;
    if(outputBuffer.size() < dest_size) outputBuffer.resize(dest_size);

    // compression 1-9, 6 is zlib default, 0 none
    int ret = compress2(&outputBuffer[0], &dest_size, inputBuffer,
			inputSize, compressionLevel);

    // check status
    if(ret == Z_OK)
      {
	// return the correct length
	resultSize = dest_size;

	FDEBUG(1) << " original size = " << inputSize
		  << " final size = " << dest_size
		  << " ratio = " << double(dest_size)/double(inputSize)
		  << endl;
      }
    else
      {
        // compression failed, return a size of zero
        FDEBUG(9) <<"Compression Return value: "<<ret
		  << " Okay = " << Z_OK << endl;
        // do we throw an exception here?
        cerr <<"Compression Return value: "<<ret<< " Okay = " << Z_OK << endl;
      }

    return resultSize;
  }
}
