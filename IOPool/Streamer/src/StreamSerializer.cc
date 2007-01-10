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

#include "zlib.h"

using namespace std;

namespace edm
{

  /**
   * Creates a translator instance for the specified product registry.
   */
  StreamSerializer::StreamSerializer(OutputModule::Selections const* selections):
    selections_(selections)
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

    for(;i!=e;++i)  
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
   */
  int StreamSerializer::serializeEvent(EventPrincipal const& eventPrincipal,
                                       EventMsgBuilder& eventMessage,
                                       bool use_compression, int compression_level)
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
      EventPrincipal::SharedConstGroupPtr const group = eventPrincipal.getGroup(id);
      if (group.get() == 0) {
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
        se.prods_.push_back(ProdPair(p, &group->provenance()));
      } else {
        se.prods_.push_back(ProdPair(group->product(), &group->provenance()));
      }
     }


    TBuffer rootbuf(TBuffer::kWrite,eventMessage.bufferSize(),
                    eventMessage.eventAddr(),kFALSE);
    RootDebug tracer(10,10);

    TClass* tc = getTClass(typeid(SendEvent));
    int bres = rootbuf.WriteObjectAny(&se,tc);
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
     
    eventMessage.setEventLength(rootbuf.Length()); 
    // compress before return if we need to
    // should test if compressed already - should never be?
    //   as double compression can have problems
    if(use_compression)
    {
      std::vector<unsigned char> dest;
      //unsigned long dest_size = 7008*1000; //(should be > rootbuf.Length()*1.001 + 12)
      unsigned long dest_size = (unsigned long)(double(rootbuf.Length())*1.002 + 1.0) + 12;
      FDEBUG(10) << "rootbuf size = " << rootbuf.Length() << " dest_size = "
           << dest_size << endl;
      dest.resize(dest_size);
      // compression 1-9, 6 is zlib default, 0 none
      int ret = compress2(&dest[0], &dest_size, (unsigned char*)eventMessage.eventAddr(),
                          rootbuf.Length(), compression_level); 
      if(ret == Z_OK) {
        // copy compressed data back into buffer and resize
        unsigned char* pos = (unsigned char*) eventMessage.eventAddr();
        unsigned char* from = (unsigned char*) &dest[0];
        unsigned int oldsize = rootbuf.Length();
        copy(from,from+dest_size,pos);
        eventMessage.setEventLength(dest_size);
        // and set reserved to original size for test and needed for buffer size
        eventMessage.setReserved(oldsize);
        // return the correct length
        FDEBUG(10) << " original size = " << oldsize << " final size = " << dest_size
             << " ratio = " << double(dest_size)/double(oldsize) << endl;
        return dest_size;
      }
      else
      {
        // compression failed, just return the original buffer
        FDEBUG(9) <<"Compression Return value: "<<ret<< " Okay = " << Z_OK << endl;
        // do we throw an exception here?
        cerr <<"Compression Return value: "<<ret<< " Okay = " << Z_OK << endl;
        return rootbuf.Length();
      }
    }
    else
    {
      // just return the original buffer
      return rootbuf.Length();
    }
  }
}
