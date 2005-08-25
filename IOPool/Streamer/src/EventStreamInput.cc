/*----------------------------------------------------------------------
$Id$
----------------------------------------------------------------------*/

#include "IOPool/Streamer/interface/EventStreamInput.h"
#include "IOPool/Streamer/interface/StreamedProducts.h"
#include "IOPool/Streamer/interface/ClassFiller.h"

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/InputServiceDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

using namespace std;

namespace edm
{

  EventStreamInput::EventStreamInput(ParameterSet const& pset,
				     InputServiceDescription const& desc) :
    InputService(desc),
    buffer_size_(pset.getParameter<int>("buffer_size")),
    event_buffer_(buffer_size_),
    file_(pset.getParameter<std::string>("fileName")),
    ist_(file_.c_str(),ios_base::binary | ios_base::in),
    store_()
  {
    if(!ist_)
      {
	throw cms::Exception("Configuration","TestProducer")
	  << "cannot open file " << file_;
      }

    loadExtraClasses();
    init();
  }

  void EventStreamInput::init()
  {
    ProductRegistry& pr = productRegistry();
    std::vector<char> inbuf(100*1000);
    TClass* prog_reg = getTClass(typeid(SendDescs));
    int len;

    // we must first read the list of product in from the front of the file and
    // add them to the desc and load there dictionaries and other stuff

    // look out for byte ordering here - this is really a test

    ist_.read((char*)&len,sizeof(int));
    ist_.read(&inbuf[0],len);

    TBuffer rootbuf(TBuffer::kRead,inbuf.size(),&inbuf[0],kFALSE);
    auto_ptr<SendDescs> sd((SendDescs*)rootbuf.ReadObjectAny(prog_reg));

    if(sd.get()==0)
      {
	throw cms::Exception("Init","ReadProductList")
	  << "Could not read the initial product registry list\n";
      }

    SendDescs::iterator i(sd->begin()),e(sd->end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct
    //cout << "Product List: " << endl;
    for(;i!=e;++i)
      {
	//cout << " " << i->fullClassName_ << endl;
	pr.copyProduct(*i);
      }

    fillStreamers(pr);

    send_event_ = getTClass(typeid(SendEvent));
  }

  EventStreamInput::~EventStreamInput()
  {
  }

  // read() is responsible for creating, and setting up, the
  // EventPrincipal.
  //
  //   1. create an EventPrincipal with a unique CollisionID
  //   2. put in one Group, holding the Provenance for the EDProduct
  //      it would be associated with.
  //   3. set up the caches in the EventPrincipal to know about this
  //      Group.
  //
  // In the future, we will *not* create the EDProduct instance (the equivalent
  // of reading the branch containing this EDProduct. That will be done by the
  // Retriever, when it is asked to do so.
  //

  auto_ptr<EventPrincipal>
  EventStreamInput::read()
  {
    int len;

    // we must first read the list of product in from the front of the file and
    // add them to the desc and load there dictionaries and other stuff

    // look out for byte ordering here - this is really a test

    ist_.read((char*)&len,sizeof(int));
    if(!ist_ || len==0) return auto_ptr<EventPrincipal>();
    ist_.read(&event_buffer_[0],len);

    TBuffer rootbuf(TBuffer::kRead,event_buffer_.size(),
		    &event_buffer_[0],kFALSE);
    auto_ptr<SendEvent> sd((SendEvent*)rootbuf.ReadObjectAny(send_event_));

    if(sd.get()==0)
      {
	throw cms::Exception("EventInput","Read")
	  << "got a null event from input stream\n";
      }

    //cout << "Got event: " << sd->id_ << endl;

    auto_ptr<EventPrincipal> ep(new EventPrincipal(sd->id_,
						   sd->time_,
						   store_,
						   productRegistry()));
    // no process name list handling

    SendProds::iterator spi(sd->prods_.begin()),spe(sd->prods_.end());
    for(;spi!=spe;++spi)
      {
	//cerr << "check prodpair" << endl;
	if(spi->prov()==0)
	  throw cms::Exception("NoData","EmptyProvenance");
	if(spi->prod()==0)
	  throw cms::Exception("NoData","EmptyProduct");
	if(spi->desc()==0)
	  throw cms::Exception("NoData","EmptyDesc");

#if 0
	cerr << "Prov:"
	     << " " << spi->desc()->fullClassName_
	     << " " << spi->prod()->id()
	     << " " << spi->desc()->productID_
	     << " " << spi->prov()->productID_
	     << endl;
#endif

	auto_ptr<EDProduct> 
	  aprod(const_cast<EDProduct*>(spi->prod()));
	auto_ptr<EventProductDescription> 
	  aedesc(const_cast<EventProductDescription*>(spi->prov()));
	auto_ptr<ProductDescription> 
	  adesc(const_cast<ProductDescription*>(spi->desc()));

	auto_ptr<Provenance> aprov(new Provenance);
	aprov->event   = *(aedesc.get());
	aprov->product = *(adesc.get());
	
	//cerr << "addgroup next" << endl;
	ep->addGroup( auto_ptr<Group>(new Group(aprod,aprov)) );
	//cerr << "addgroup done" << endl;
	spi->clear();
      }

    return ep;
  }


  EventStreamInput::StreamRetriever::~StreamRetriever() {}

  auto_ptr<EDProduct>
  EventStreamInput::StreamRetriever::get(BranchKey const& k)
  {
    throw cms::Exception("LogicError","StreamRetriever")
      << "Got into EventStreamInput::StreamRetriever for branchkey: "
      << k << "\n";

    return auto_ptr<EDProduct>();
  }
}
