
//////////////////////////////////////////////////////////////////////
//
// $Id: EventStreamOutput.cc,v 1.6 2005/10/11 21:34:29 wmtan Exp $
//
// Class EventStreamOutput module
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Framework/interface/Provenance.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "IOPool/Streamer/interface/StreamedProducts.h"
#include "IOPool/Streamer/interface/ClassFiller.h"

#include "PluginManager/PluginCapabilities.h"
#include "SealBase/SharedLibrary.h"

#include "StorageSvc/IOODatabaseFactory.h"
#include "StorageSvc/IClassLoader.h"
#include "StorageSvc/DbType.h"

#include "TBuffer.h"
#include "TClass.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <list>

using namespace std;

namespace edm
{
  void event_stream_output_test_read(void* buf, int len, TClass* send_event_)
  {
    TBuffer rootbuf(TBuffer::kRead,len,buf,kFALSE);
    if(10<debugit()) gDebug=10;
    auto_ptr<SendEvent> sd((SendEvent*)rootbuf.ReadObjectAny(send_event_));
    if(10<debugit()) gDebug=0;

    if(sd.get()==0)
      {
	throw cms::Exception("DecodeEvent","test_read")
	  << "got a null event from input stream\n";
      }

    cout << "Got event: " << sd->id_ << endl;

    SendProds::iterator spi(sd->prods_.begin()),spe(sd->prods_.end());
    for(;spi!=spe;++spi)
      {
	if(spi->prov()==0)
	  throw cms::Exception("NoData","EmptyProvenance");
	if(spi->prod()==0)
	  throw cms::Exception("NoData","EmptyProduct");
	if(spi->desc()==0)
	  throw cms::Exception("NoData","EmptyDesc");

	cout << "Prov:"
	     << " " << spi->desc()->fullClassName_
	     << " " << spi->desc()->productID_
	     << " " << spi->prov()->productID_
	     << endl;

      }
  }

  // -------------------------------------

  EventStreamerImpl::EventStreamerImpl(ParameterSet const&,
				       EventBuffer* bufs):
    bufs_(bufs),
    tc_(),
    prod_reg_buf_(100 * 1000),
    prod_reg_len_()
  {
    FDEBUG(6) << "StreamOutput constructor" << endl;
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
    FDEBUG(6) << "StreamOutput: serializeRegistry" << endl;
    TClass* prog_reg = getTClass(typeid(SendJobHeader));
    SendJobHeader sd;

    Selections::const_iterator i(prods.begin()),e(prods.end());

    //cout << "Product List: " << endl;
    for(;i!=e;++i) 
      {
	//cout << " " << (*i)->fullClassName_ << endl;
	sd.descs_.push_back(**i);
	FDEBUG(9) << "StreamOutput got product = " << (*i)->fullClassName_
		  << endl;
      }

    TBuffer rootbuf(TBuffer::kWrite,
		    prod_reg_buf_.size(),&prod_reg_buf_[0],
		    kFALSE);
    if(10<debugit()) gDebug=10;
    int bres = rootbuf.WriteObjectAny((char*)&sd,prog_reg);
    if(10<debugit()) gDebug=0;
    
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
    prod_reg_len_ = rootbuf.Length();
  }

  void EventStreamerImpl::serialize(EventPrincipal const& e)
  {
    // all provenance data needs to be transferred, including the
    // indirect stuff referenced from the product provenance structure.
    SendEvent se(e.id(),e.time());

    EventPrincipal::const_iterator i(e.begin()),ie(e.end());
    for(;i!=ie; ++i)
      {
	const Group* group = (*i).get();
	if (true) // selected(group->provenance().product))
	  {
	    // necessary for now - will need to be improved
	    if (group->product()==0)
	      e.get(group->provenance().product.productID_);

#if 0
	    cout << "Prov:"
		 << " " << group->provenance().product.fullClassName_
		 << " " << group->provenance().product.productID_
		 << endl;
#endif

	    se.prods_.push_back(
				ProdPair(group->product(),
					 &group->provenance())
				);


	  }
      }	

#if 0
    cout << "-----Dump start" << endl;
    for(SendProds::iterator pii=se.prods_.begin();pii!=se.prods_.end();++pii)
      cout << "Prov:"
	   << " " << pii->desc()->fullClassName_
	   << " " << pii->desc()->productID_
	   << endl;      
    cout << "-----Dump end" << endl;
#endif

    EventBuffer::ProducerBuffer b(*bufs_);
    //gDebug=10;
    TBuffer rootbuf(TBuffer::kWrite,b.size(),b.buffer(),kFALSE);
    if(10<debugit()) gDebug=10;
    int bres = rootbuf.WriteObjectAny(&se,tc_);
    if(10<debugit()) gDebug=0;
    
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
     
    b.commit(rootbuf.Length());
    // test_read(b.buffer(),b.size(),tc_);
  }


}
