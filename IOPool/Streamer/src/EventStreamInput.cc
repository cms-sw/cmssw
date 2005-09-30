/*----------------------------------------------------------------------
$Id: EventStreamInput.cc,v 1.9 2005/09/28 05:38:11 wmtan Exp $
----------------------------------------------------------------------*/

#include "IOPool/Streamer/interface/EventStreamInput.h"
#include "IOPool/Streamer/interface/StreamedProducts.h"
#include "IOPool/Streamer/interface/ClassFiller.h"

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <cassert>

using namespace std;

namespace edm
{

  EventStreamerInputImpl::EventStreamerInputImpl(ParameterSet const&,
						 InputSourceDescription const& desc,
						 EventBuffer* bufs) :
    regbuf_(1000*1000),
    bufs_(bufs),
    pr_(desc.preg_),
    send_event_()
  {
    loadExtraClasses();
    init();
  }

  void EventStreamerInputImpl::init()
  {
  }

  void EventStreamerInputImpl::decodeRegistry()
  {
    FDEBUG(6) << "StreamInput: decodeRegistry" << endl;

    TClass* prog_reg = getTClass(typeid(SendJobHeader));
    TBuffer rootbuf(TBuffer::kRead,regbuf_.size(),&regbuf_[0],kFALSE);
    if(10<debugit()) gDebug=10;
    auto_ptr<SendJobHeader> 
      sd((SendJobHeader*)rootbuf.ReadObjectAny(prog_reg));
    if(10<debugit()) gDebug=0;

    if(sd.get()==0)
      {
	throw cms::Exception("Init","DecodeProductList")
	  << "Could not read the initial product registry list\n";
      }

    SendDescs::iterator i(sd->descs_.begin()),e(sd->descs_.end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct
    //cout << "Product List: " << endl;
    for(;i!=e;++i)
      {
	//cout << " " << i->fullClassName_ << endl;
	pr_->copyProduct(*i);
	FDEBUG(6) << "StreamInput product = " << i->fullClassName_ << endl;
      }

    // fillStreamers(*pr_);

    // this is not good - delay setting send_event_ until now,
    // so this class cannot be used properly without the header
    send_event_ = getTClass(typeid(SendEvent));
  }

  EventStreamerInputImpl::~EventStreamerInputImpl()
  {
  }

  // reconstitute() is responsible for creating, and setting up, the
  // EventPrincipal.
  //
  // All products are reconstituted.
  // 
  //

  auto_ptr<EventPrincipal>
  EventStreamerInputImpl::reconstitute()
  {
    assert(send_event_!=0);

    EventBuffer::ConsumerBuffer pb(*bufs_);

    if(pb.size()==0) return auto_ptr<EventPrincipal>();

    TBuffer rootbuf(TBuffer::kRead,pb.size(),
		    (char*)pb.buffer(),kFALSE);
    if(10<debugit()) gDebug=10;
    auto_ptr<SendEvent> sd((SendEvent*)rootbuf.ReadObjectAny(send_event_));
    if(10<debugit()) gDebug=0;

    if(sd.get()==0)
      {
	throw cms::Exception("EventInput","Read")
	  << "got a null event from input stream\n";
      }

    //cout << "Got event: " << sd->id_ << endl;

    auto_ptr<EventPrincipal> ep(new EventPrincipal(sd->id_,
						   sd->time_,
						   *pr_));
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
	auto_ptr<BranchEntryDescription> 
	  aedesc(const_cast<BranchEntryDescription*>(spi->prov()));
	auto_ptr<ProductDescription> 
	  adesc(const_cast<ProductDescription*>(spi->desc()));

	auto_ptr<Provenance> aprov(new Provenance);
	aprov->event   = *(aedesc.get());
	aprov->product = *(adesc.get());
	
	//cerr << "addgroup next" << endl;
	ep->addGroup(auto_ptr<Group>(new Group(aprod,aprov)));
	//cerr << "addgroup done" << endl;
	spi->clear();
      }

    return ep;
  }
}
