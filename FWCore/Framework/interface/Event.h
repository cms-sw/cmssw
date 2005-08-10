#ifndef EDM_EVENT_HH
#define EDM_EVENT_HH

/*----------------------------------------------------------------------

Event: This is the primary interface for accessing
EDProducts from a single collision and inserting new derived products.

$Id: Event.h,v 1.13 2005/07/30 23:44:24 wmtan Exp $

----------------------------------------------------------------------*/
#include <cassert>
#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

#include "FWCore/EDProduct/interface/EventID.h"
#include "FWCore/EDProduct/interface/Timestamp.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/BasicHandle.h"

#include "FWCore/Framework/src/Group.h"
#include "FWCore/Framework/src/TypeID.h"

namespace edm {

  class Event
  {
  public:
    Event(EventPrincipal& ep, const ModuleDescription&);
    ~Event();

    // these must come from the ep.
    EventID id() const;
    Timestamp time() const;
    
    // How do these get set in the first place?
    const LuminositySection& getLuminositySection() const;
    const Run& getRun() const;

    template <class PROD>
    void 
    put(std::auto_ptr<PROD> product);

    template <class PROD>
    void 
    put(std::auto_ptr<PROD> product, const std::string& productInstanceName);

    template <class PROD>
    void 
    get(ProductID id, Handle<PROD>& result) const;

    template <class PROD>
    void 
    get(const Selector&, Handle<PROD>& result) const;
  
    template <class PROD>
    void 
    getByLabel(const std::string& label, Handle<PROD>& result) const;

    template <class PROD>
    void 
    getByLabel(const std::string& label, const std::string& productInstanceName, Handle<PROD>& result) const;

    template <class PROD>
    void 
    getMany(const Selector&, std::vector<Handle<PROD> >& results) const;

    const Provenance* get(ProductID id) const;

  private:
    typedef std::vector<ProductID>       ProductIDVec;
    //typedef std::vector<const Group*> GroupPtrVec;
    typedef std::vector<std::pair<EDProduct*, std::string> >   ProductPtrVec;
    typedef std::vector<BasicHandle>  BasicHandleVec;    

    //------------------------------------------------------------
    // Private functions.
    //
    // commit() is called to complete the transaction represented by
    // this Event. The friendship required seems gross, but any
    // alternative is not great either.  and putting it into the
    // public interface is asking for trouble
    void commit_();
    friend class ProducerWorker;

    // The following 'get' functions serve to isolate the Event class
    // from the EventPrincipal class.

    BasicHandle 
    get_(ProductID oid) const;

    BasicHandle 
    get_(TypeID id, const Selector&) const;
    
    BasicHandle 
    getByLabel_(TypeID id,
		const std::string& label,
		const std::string& productInstanceName) const;

    void 
    getMany_(TypeID id, 
	     const Selector& sel, 
	     BasicHandleVec& results) const;

    //------------------------------------------------------------
    // Copying and assignment of Events is disallowed
    //
    Event(const Event&);                  // not implemented
    const void operator=(const Event&);   // not implemented

    //------------------------------------------------------------
    // Data members
    //

    // put_products_ is the holding pen for EDProducts inserted into
    // this Event. Pointers in this collection own the products to
    // which they point.
    ProductPtrVec            put_products_;

    // gotProductIDs_ must be mutable because it records all 'gets',
    // which do not logically modify the Event. gotProductIDs_ is
    // merely a cache reflecting what has been retreived from the
    // EventPrincipal.
    mutable ProductIDVec        gotProductIDs_;

    // Each Event must have an associated EventPrincipal, used as the
    // source of all 'gets' and the target of 'puts'.
    EventPrincipal&          ep_;

    // Each Event must have a description of the module executing the
    // "transaction" which the Event represents.
    const ModuleDescription& md_;
  };


  //------------------------------------------------------------
  //
  // Utilities for creation of handles.
  //
  
  template <class PROD>
  Handle<PROD> make_handle(const Group& g)
  {
    return Handle<PROD>(g.product(), g.provenance());
  }
 
  template <class PROD>
  struct makeHandle
  {
    Handle<PROD> operator()(const Group& g) { return make_handle<PROD>(g); }
    Handle<PROD> operator()(const Group* g) { return make_handle<PROD>(*g); }
  };


  //------------------------------------------------------------
  //
  // Implementation of  Event  member templates. See  Event.cc for the
  // implementation of non-template members.
  //

  template <class PROD>
  inline
  void 
  Event::put(std::auto_ptr<PROD> product)
  {
    put(product, std::string());
  }

  template <class PROD>
  void 
  Event::put(std::auto_ptr<PROD> product, const std::string& productInstanceName)
  {
    PROD* p = product.get();
    assert (p);                // null pointer is illegal
    edm::Wrapper<PROD> *wp(new Wrapper<PROD>(*p));
    put_products_.push_back(std::make_pair(wp, productInstanceName));
    product.release();
  }

  template <class PROD>
  void
  Event::get(ProductID id, Handle<PROD>& result) const
  {
    BasicHandle bh = this->get_(TypeID(typeid(PROD)), id);
    gotProductIDs_.push_back(bh.id());
    convert_handle(bh, result);  // thrown on conversion error
  }

  template <class PROD>
  void 
  Event::get(const Selector& sel,
	     Handle<PROD>& result) const
  {
    BasicHandle bh = this->get_(TypeID(typeid(PROD)),sel);
    gotProductIDs_.push_back(bh.id());
    convert_handle(bh, result);  // thrown on conversion error
  }
  
  template <class PROD>
  inline
  void
  Event::getByLabel(const std::string& label,
		    Handle<PROD>& result) const
  {
    getByLabel(label, std::string(), result);
  }

  template <class PROD>
  void
  Event::getByLabel(const std::string& label,
                    const std::string& productInstanceName,
		    Handle<PROD>& result) const
  {
    BasicHandle bh = this->getByLabel_(TypeID(typeid(PROD)), label, productInstanceName);
    gotProductIDs_.push_back(bh.id());
    convert_handle(bh, result);  // thrown on conversion error
  }

  template <class PROD>
  void 
  Event::getMany(const Selector& sel,
		 std::vector<Handle<PROD> >& results) const
  { 
    BasicHandleVec bhv;
    this->getMany_(TypeID(typeid(PROD)), sel, bhv);
    
    // Go through the returned handles; for each element,
    //   1. create a Handle<PROD> and
    //   2. record the ProductID in gotProductIDs
    //
    // This function presents an exception safety difficulty. If an
    // exception is thrown when converting a handle, the "got
    // products" record will be wrong.
    //
    // Since EDProducers are not allowed to use this function,
    // the problem does not seem too severe.
    //
    // Question: do we even need to keep track of the "got products"
    // for this function, since it is *not* to be used by EDProducers?
    std::vector<Handle<PROD> > products;

    BasicHandleVec::const_iterator it = bhv.begin();
    BasicHandleVec::const_iterator end = bhv.end();

    while (it != end)
      {
	gotProductIDs_.push_back((*it).id());
	Handle<PROD> result;
	convert_handle(*it, result);  // thrown on conversion error
	products.push_back(result);
	++it;
      }
    results.swap(products);
  }
}
#endif
