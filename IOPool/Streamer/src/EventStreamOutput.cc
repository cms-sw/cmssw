
//////////////////////////////////////////////////////////////////////
//
// $Id: EventStreamOutput.cc,v 1.29 2007/06/29 16:41:23 wmtan Exp $
//
// Class EventStreamOutput module
//
//////////////////////////////////////////////////////////////////////

#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "IOPool/Streamer/interface/Messages.h"
#include "IOPool/Streamer/interface/ClassFiller.h"

#include "TBuffer.h"
#include "TClass.h"

#include <memory>
#include <string>
#include <list>

namespace edm
{
  void event_stream_output_test_read(void* buf, int len, TClass* send_event_)
  {
    EventMsg msg(buf,len);
    TBuffer rootbuf(TBuffer::kRead,msg.getDataSize(),msg.data(),kFALSE);

    RootDebug tracer(10,10);
    std::auto_ptr<SendEvent> sd((SendEvent*)rootbuf.ReadObjectAny(send_event_));

    if(sd.get()==0)
      {
	throw cms::Exception("DecodeEvent","test_read")
	  << "got a null event from input stream\n";
      }

    std::cout << "Got event: " << sd->id_ << std::endl;

    SendProds::iterator spi(sd->prods_.begin()),spe(sd->prods_.end());
    for(; spi != spe; ++spi)
      {
	if(spi->prov() == 0)
	  throw cms::Exception("NoData","EmptyProvenance");
	if(spi->prod() == 0)
	  throw cms::Exception("NoData","EmptyProduct");
	if(spi->desc() == 0)
	  throw cms::Exception("NoData","EmptyDesc");

	FDEBUG(2) << "Prov:"
	     << " " << spi->desc()->className()
	     << " " << spi->desc()->productID()
	     << " " << spi->prov()->productID_
	     << std::endl;

      }
  }

  // -------------------------------------

  EventStreamerImpl::EventStreamerImpl(ParameterSet const&,
				       Selections const* selections,
				       EventBuffer* bufs):
    selections_(selections),
    bufs_(bufs),
    tc_(),
    prod_reg_buf_(100 * 1000),
    prod_reg_len_()
  {
    FDEBUG(6) << "StreamOutput constructor" << std::endl;
    // unsure whether or not we need to do the declareStreamer call or not
    //fillStreamers(reg);
    loadExtraClasses();
    tc_ = getTClass(typeid(SendEvent));
    // serializeRegistry(reg); // now called directed from beginJob()
  }

  EventStreamerImpl::~EventStreamerImpl()
  {
  }

  void EventStreamerImpl::serializeRegistry(Selections const& prods)
  {
    FDEBUG(6) << "StreamOutput: serializeRegistry" << std::endl;
    TClass* prog_reg = getTClass(typeid(SendJobHeader));
    SendJobHeader sd;

    Selections::const_iterator i(prods.begin()),e(prods.end());

    FDEBUG(9) << "Product List: " << std::endl;
    for(; i != e; ++i) {
	sd.descs_.push_back(**i);
	FDEBUG(9) << "StreamOutput got product = " << (*i)->className()
		  << std::endl;
    }

    InitMsg im(&prod_reg_buf_[0],prod_reg_buf_.size(),true);
    TBuffer rootbuf(TBuffer::kWrite,im.getDataSize(),im.data(),kFALSE);
    RootDebug tracer(10,10);

    int bres = rootbuf.WriteObjectAny((char*)&sd,prog_reg);
    
    switch(bres)
      {
      case 0: // failure
	{
	  throw cms::Exception("Output","SerializationReg")
	    << "EventStreamOutput module could not serialize event\n";
	  break;
	}
      case 1: // succcess
	break;
      case 2: // truncated result
	{
	  throw cms::Exception("Output","SerializationReg")
	    << "EventStreamOutput module attempted to serialize the registry\n"
	    << "that is to big for the allocated buffers\n";
	  break;
	}
      default: // unknown
	{
	  throw cms::Exception("Output","SerializationReg")
	    << "EventStreamOutput module got an unknown error code\n"
	    << " while attempting to serialize registry\n";
	  break;
	}
      }	    
    im.setDataSize(rootbuf.Length());
    prod_reg_len_ = im.msgSize();
  }

  void EventStreamerImpl::serialize(EventPrincipal const& e)
  {
    // all provenance data needs to be transferred, including the
    // indirect stuff referenced from the product provenance structure.
    SendEvent se(e.id(),e.time());

    Selections::const_iterator i(selections_->begin()),ie(selections_->end());
    // Loop over EDProducts, fill the provenance, and write.
    for(; i != ie; ++i) {
      BranchDescription const& desc = **i;
      ProductID const& id = desc.productID();

      if (id == ProductID()) {
	throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	  << "EventStreamOutput::serialize: invalid ProductID supplied in productRegistry\n";
      }
      BasicHandle const bh = e.getForOutput(id, true);
      assert(bh.provenance());
      // ModuleDescription md = group->provenance().moduleDescription();
      if (bh.wrapper() == 0) {
        std::string const& name = desc.className();
        std::string const className = wrappedClassName(name);
        TClass *cp = gROOT->GetClass(className.c_str());
        if (cp == 0) {
          throw edm::Exception(errors::ProductNotFound,"NoMatch")
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

#if 0
    FDEBUG(11) << "-----Dump start" << std::endl;
    for(SendProds::iterator pii = se.prods_.begin(), piiEnd = se.prods_.end(); pii != piiEnd; ++pii)
      std::cout << "Prov:"
	   << " " << pii->desc()->className()
	   << " " << pii->desc()->productID()
	   << std::endl;      
    FDEBUG(11) << "-----Dump end" << std::endl;
#endif

    EventBuffer::ProducerBuffer b(*bufs_);
    EventMsg msg(b.buffer(),b.size(),e.id().event(),e.id().run(),1,1);
    TBuffer rootbuf(TBuffer::kWrite,msg.getDataSize(),msg.data(),kFALSE);
    RootDebug tracer(10,10);

    int bres = rootbuf.WriteObjectAny(&se,tc_);
    
    switch(bres)
      {
      case 0: // failure
	{
	  throw cms::Exception("Output","Serialization")
	    << "EventStreamOutput module could not serialize event: "
	    << e.id();
	  break;
	}
      case 1: // succcess
	break;
      case 2: // truncated result
	{
	  throw cms::Exception("Output","Serialization")
	    << "EventStreamOutput module attempted to serialize an event\n"
	    << "that is to big for the allocated buffers: "
	    << e.id();
	  break;
	}
      default: // unknown
	{
	  throw cms::Exception("Output","Serialization")
	    << "EventStreamOutput module got an unknown error code\n"
	    << " while attempting to serialize event: "
	    << e.id();
	  break;
	}
      }
     
    msg.setDataSize(rootbuf.Length());
    b.commit(msg.msgSize());
    // test_read(b.buffer(),b.size(),tc_);
  }


}
