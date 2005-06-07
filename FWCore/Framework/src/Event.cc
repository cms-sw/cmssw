/*----------------------------------------------------------------------
$Id: Event.cc,v 1.2 2005/06/03 04:04:47 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <algorithm>

#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/EDProduct/interface/EDP_ID.h"
#include "FWCore/CoreFramework/interface/EventPrincipal.h"
#include "FWCore/CoreFramework/src/Group.h"

using namespace std;

namespace edm {

  Event::Event(EventPrincipal& ep, const ModuleDescription& md) :
    ep_(ep),
    md_(md)
  {  }

  struct deleter
  {
    void operator()(const EDProduct* p) const { delete p; }
  };

  Event::~Event()
  {
    // anything left here must be the result of a failure
    // let's record them as failed attempts in the event principal
    std::for_each(put_products_.begin(),put_products_.end(),deleter());

#if 0
    // this is not good enough (the code below) - it only covered part of the 
    // problem.  We must know how many thing were to be produced,
    // and there TypeID, so that we can leave around a bit of
    // provenance indicating bad status (and isacessibility)

    ProductVec::iterator i(put_products_.begin()),e(put_products_.end());

    while(i!=e)
    {
      auto_ptr<EDProduct> pr(*pit);
      // note: ownership has been past - so clear the pointer!
      *pit = 0;
      auto_ptr<Provenance> pv(new Provenance(md_));
      
      // set parts of provenance
      pv->cid = 0; // what is this supposed to be?
      // what is this supposed to be? this is a disgusting string.
      pv->full_product_type_name = TypeID(*pr).reflectionClassName();
      pv->friendly_product_type_name = TypeID(*pr).friendlyClassName();
      pv->status = Provenance::Success;
      pv->parents = idlist;

	ep_.put(pr,pv);
      
      ++i;
    }
#endif
  }

  CollisionID
  Event::ID() const
  {
    return ep_.ID();
  }

  void 
  Event::commit_()
  {
    // fill in guts of provenance here
    ProductPtrVec::iterator pit(put_products_.begin());
    ProductPtrVec::iterator pie(put_products_.end());

    while(pit!=pie)
      {
	auto_ptr<EDProduct> pr(*pit);
	// note: ownership has been passed - so clear the pointer!
	*pit = 0;
	auto_ptr<Provenance> pv(new Provenance(md_));

	// set parts of provenance
	pv->cid = 0; // TODO: what is this supposed to be?
	// TODO: what is this supposed to be? this is a disgusting string.
	pv->full_product_type_name = TypeID(*pr).reflectionClassName();
	pv->friendly_product_type_name = TypeID(*pr).friendlyClassName();
	pv->status = Provenance::Success;
	pv->parents = got_product_ids_;

	ep_.put(pr,pv);
	++pit;
      }

    // the cleanup is all or none
    put_products_.clear();
  }

  Event::BasicHandle
  Event::get_(EDP_ID oid) const
  {
    return ep_.get(oid);
  }

  Event::BasicHandle
  Event::get_(TypeID id, const Selector& sel) const
  {
    return ep_.getBySelector(id, sel);
  }
    
  Event::BasicHandle
  Event::getByLabel_(TypeID id,
		     const string& label) const
  {
    return ep_.getByLabel(id, label);
  }

  void 
  Event::getMany_(TypeID id, 
		  const Selector& sel,
		  BasicHandleVec& results) const
  {
    ep_.getMany(id, sel, results);
  }

}
