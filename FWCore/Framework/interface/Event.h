#ifndef Framework_Event_h
#define Framework_Event_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     Event
// 
/**\class Event Event.h FWCore/Framework/interface/Event.h

Description: This is the primary interface for accessing EDProducts
from a single collision and inserting new derived products.

For its usage, see "FWCore/Framework/interface/DataViewImpl.h"

*/
/*----------------------------------------------------------------------

$Id: Event.h,v 1.63.2.2 2008/05/12 15:33:07 wmtan Exp $

----------------------------------------------------------------------*/

#include <vector>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class Event : private DataViewImpl<EventEntryInfo>
  {
  public:
    typedef DataViewImpl<EventEntryInfo> Base;
    Event(EventPrincipal& ep, const ModuleDescription& md);
    ~Event(){}

    // AUX functions.
    EventID id() const {return aux_.id();}
    Timestamp time() const {return aux_.time();}
    LuminosityBlockNumber_t
    luminosityBlock() const {return aux_.luminosityBlock();}
    bool isRealData() const {return aux_.isRealData();}
    EventAuxiliary::ExperimentType experimentType() const {return aux_.experimentType();}

    using Base::get;
    using Base::getByLabel;
    using Base::getByType;
    using Base::getMany;
    using Base::getManyByType;
    using Base::me;
    using Base::processHistory;
    using Base::size;

    LuminosityBlock const&
    getLuminosityBlock() const {
      return *luminosityBlock_;
    }

    Run const&
    getRun() const;

    RunNumber_t
    run() const {return id().run();}   

    template <typename PROD>
    bool 
    getByProductID(ProductID const& oid, Handle<PROD>& result) const;

    // Template member overload to deal with Views.     
    template <typename ELEMENT>
    bool
    getByProductID(ProductID const& oid, Handle<View<ELEMENT> >& result) const ;

    History const& history() const;

    ///Put a new product.
    template <typename PROD>
    OrphanHandle<PROD>
    put(std::auto_ptr<PROD> product) {return put<PROD>(product, std::string());}

    ///Put a new product with a 'product instance name'
    template <typename PROD>
    OrphanHandle<PROD>
    put(std::auto_ptr<PROD> product, std::string const& productInstanceName);

    ///Returns a RefProd to a product before that product has been placed into the Event.
    /// The RefProd (and any Ref's made from it) will no work properly until after the
    /// Event has been committed (which happens after leaving the EDProducer::produce method)
    template <typename PROD>
    RefProd<PROD>
    getRefBeforePut() {return getRefBeforePut<PROD>(std::string());}

    template <typename PROD>
    RefProd<PROD>
    getRefBeforePut(std::string const& productInstanceName);

    template <typename PROD>
    bool 
    get(SelectorBase const& sel,
  		    Handle<PROD>& result) const;
    
    template <typename PROD>
    bool
    getByLabel(InputTag const& tag, Handle<PROD>& result) const;
  
    template <typename PROD>
    bool
    getByLabel(std::string const& label, Handle<PROD>& result) const;
  
    template <typename PROD>
    bool
    getByLabel(std::string const& label,
  			   std::string const& productInstanceName,
  			   Handle<PROD>& result) const;
  
    template <typename PROD>
    void 
    getMany(SelectorBase const& sel,
  			std::vector<Handle<PROD> >& results) const;
  
    template <typename PROD>
    bool
    getByType(Handle<PROD>& result) const;
  
    template <typename PROD>
    void 
    getManyByType(std::vector<Handle<PROD> >& results) const;

    // Template member overload to deal with Views. Perhaps only this
    // one needs to be overloaded, because the other getByLabel
    // implementations go through this one.
    template <typename ELEMENT>
    bool
    getByLabel(std::string const& label, 
	       std::string const& productInstanceName,
	       Handle<View<ELEMENT> >& result) const;

    template <typename ELEMENT> 	 
    bool 	 
    getByLabel(InputTag const& tag, Handle<View<ELEMENT> >& result) const; 	 
    
    template <typename ELEMENT>
    void
    fillView_(BasicHandle & bh,
	      Handle<View<ELEMENT> >& result) const;

    Provenance
    getProvenance(BranchID const& theID) const;

    void
    getAllProvenance(std::vector<Provenance const*> &provenances) const;

  private:
    EventPrincipal const&
    eventPrincipal() const;

    EventPrincipal &
    eventPrincipal();

    // commit_() is called to complete the transaction represented by
    // this DataViewImpl. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
    friend class DaqSource;
    friend class InputSource;
    friend class RawInputSource;
    friend class EDFilter;
    friend class EDProducer;

    void commit_();

    BasicHandle 
    getByProductID_(ProductID const& oid) const;

    EventAuxiliary const& aux_;
    boost::shared_ptr<LuminosityBlock const> const luminosityBlock_;

    // gotProductIDs_ must be mutable because it records all 'gets',
    // which do not logically modify the DataViewImpl. gotProductIDs_ is
    // merely a cache reflecting what has been retreived from the
    // Principal class.
    typedef std::vector<ProductID> ProductIDVec;
    mutable ProductIDVec gotProductIDs_;

    // We own the retrieved Views, and have to destroy them.
    mutable std::vector<boost::shared_ptr<ViewBase> > gotViews_;
  };

  template <typename PROD>
  bool
  Event::getByProductID(ProductID const& oid, Handle<PROD>& result) const
  {
    result.clear();
    BasicHandle bh = this->getByProductID_(oid);
    convert_handle(bh, result);  // throws on conversion error
    if (bh.failedToGet()) {
      return false;
    }
    gotProductIDs_.push_back(bh.id());
    return true;
  }

  template <typename ELEMENT>
  bool
  Event::getByProductID(ProductID const& oid, Handle<View<ELEMENT> >& result) const
  {
      result.clear();
      BasicHandle bh = this->getByProductID_(oid);

      if(bh.failedToGet()) {
          boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound) );
          *whyFailed
              << "get View by ID failed: no product with ID = " << oid.id() <<"\n";
          Handle<View<ELEMENT> > temp(whyFailed);
          result.swap(temp);
          return false;
      }

      fillView_(bh, result);
      return true;
  }

  template <typename PROD>
  OrphanHandle<PROD> 
  Event::put(std::auto_ptr<PROD> product, std::string const& productInstanceName)
  {
    if (product.get() == 0) {                // null pointer is illegal
      TypeID typeID(typeid(PROD));
      throw edm::Exception(edm::errors::NullPointerError)
        << "Event::put: A null auto_ptr was passed to 'put'.\n"
	<< "The pointer is of type " << typeID << ".\n"
	<< "The specified productInstanceName was '" << productInstanceName << "'.\n";
    }

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    typename boost::mpl::if_c<detail::has_postinsert<PROD>::value, 
      DoPostInsert<PROD>, 
      DoNotPostInsert<PROD> >::type maybe_inserter;
    maybe_inserter(product.get());

    ConstBranchDescription const& desc =
      getBranchDescription(TypeID(*product), productInstanceName);

    Wrapper<PROD> *wp(new Wrapper<PROD>(product));

    putProducts().push_back(std::make_pair(wp, &desc));

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.

    return(OrphanHandle<PROD>(wp->product(), desc.productIDtoAssign()));
  }

  template <typename PROD>
  RefProd<PROD>
  Event::getRefBeforePut(std::string const& productInstanceName) {
    PROD* p = 0;
    ConstBranchDescription const& desc =
      getBranchDescription(TypeID(*p), productInstanceName);

    //should keep track of what Ref's have been requested and make sure they are 'put'
    return RefProd<PROD>(desc.productIDtoAssign(), prodGetter());
  }

  template <typename PROD>
  bool 
  Event::get(SelectorBase const& sel,
		    Handle<PROD>& result) const
  {
    bool ok = this->Base::get(sel, result);
    if (ok) {
      gotProductIDs_.push_back(result.id());
    }
    return ok;
  }
  
  template <typename PROD>
  bool
  Event::getByLabel(InputTag const& tag, Handle<PROD>& result) const
  {
    bool ok = this->Base::getByLabel(tag, result);
    if (ok) {
      gotProductIDs_.push_back(result.id());
    }
    return ok;
  }

  template <typename PROD>
  bool
  Event::getByLabel(std::string const& label, Handle<PROD>& result) const
  {
    bool ok = this->Base::getByLabel(label, result);
    if (ok) {
      gotProductIDs_.push_back(result.id());
    }
    return ok;
  }

  template <typename PROD>
  bool
  Event::getByLabel(std::string const& label,
			   std::string const& productInstanceName,
			   Handle<PROD>& result) const
  {
    bool ok = this->Base::getByLabel(label, productInstanceName, result);
    if (ok) {
      gotProductIDs_.push_back(result.id());
    }
    return ok;
  }

  template <typename PROD>
  void 
  Event::getMany(SelectorBase const& sel,
			std::vector<Handle<PROD> >& results) const
  { 
    this->Base::getMany(sel, results);
    for (typename std::vector<Handle<PROD> >::const_iterator it = results.begin(), itEnd = results.end();
        it != itEnd; ++it) {
      gotProductIDs_.push_back(it->id());
    }
  }

  template <typename PROD>
  bool
  Event::getByType(Handle<PROD>& result) const
  {
    bool ok = this->Base::getByType(result);
    if (ok) {
      gotProductIDs_.push_back(result.id());
    }
    return ok;
  }

  template <typename PROD>
  void 
  Event::getManyByType(std::vector<Handle<PROD> >& results) const
  { 
    this->Base::getManyByType(results);
    for (typename std::vector<Handle<PROD> >::const_iterator it = results.begin(), itEnd = results.end();
        it != itEnd; ++it) {
      gotProductIDs_.push_back(it->id());
    }
  }
  
  template <typename ELEMENT>
  bool
  Event::getByLabel(std::string const& moduleLabel,
			   std::string const& productInstanceName,
			   Handle<View<ELEMENT> >& result) const
  {
   result.clear();

    TypeID typeID(typeid(ELEMENT));

    BasicHandleVec bhv;
    int nFound = getMatchingSequenceByLabel_(typeID,
                                             moduleLabel,
                                             productInstanceName,
                                             bhv,
                                             true);

    if (nFound == 0) {
      boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound) );
      *whyFailed
	<< "getByLabel: Found zero products matching all criteria\n"
	<< "Looking for sequence of type: " << typeID << "\n"
	<< "Looking for module label: " << moduleLabel << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n";
      Handle<View<ELEMENT> > temp(whyFailed);
      result.swap(temp);
      return false;
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getByLabel: Found more than one product matching all criteria\n"
	<< "Looking for sequence of type: " << typeID << "\n"
	<< "Looking for module label: " << moduleLabel << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n";
    }

    fillView_(bhv[0], result);
    return true;
  }

  template <typename ELEMENT>
    bool
    Event::getByLabel(InputTag const& tag, Handle<View<ELEMENT> >& result) const
  {
    result.clear();
    if (tag.process().empty()) {
      return getByLabel(tag.label(), tag.instance(), result);
    } else {
      TypeID typeID(typeid(ELEMENT));
      
      BasicHandleVec bhv;
      int nFound = getMatchingSequenceByLabel_(typeID,
                                               tag.label(),
                                               tag.instance(),
                                               tag.process(),
                                               bhv,
                                               true);
      
      if (nFound == 0) {
        boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound) );
        *whyFailed
          << "getByLabel: Found zero products matching all criteria\n"
          << "Looking for sequence of type: " << typeID << "\n"
          << "Looking for module label: " << tag.label() << "\n"
          << "Looking for productInstanceName: " << tag.instance() << "\n"
          << "Looking for processName: "<<tag.process() <<"\n";
        Handle<View<ELEMENT> > temp(whyFailed);
        result.swap(temp);
        return false;
      }
      if (nFound > 1) {
        throw edm::Exception(edm::errors::ProductNotFound)
        << "getByLabel: Found more than one product matching all criteria\n"
	<< "Looking for sequence of type: " << typeID << "\n"
        << "Looking for module label: " << tag.label() << "\n"
        << "Looking for productInstanceName: " << tag.instance() << "\n"
        << "Looking for processName: "<<tag.process() <<"\n";
      }
      
      fillView_(bhv[0], result);
      return true;
    }
    return false;
  }

  template <typename ELEMENT>
  void
  Event::fillView_(BasicHandle & bh,
			  Handle<View<ELEMENT> >& result) const
  {
    std::vector<void const*> pointersToElements;
    // the following is a shared pointer. 
    // It is not initialized here
    helper_vector_ptr helpers;
    // the following must initialize the
    //  shared pointer and fill the helper vector
    bh.wrapper()->fillView(bh.id(), pointersToElements, helpers);

    boost::shared_ptr<View<ELEMENT> > 
      newview(new View<ELEMENT>(pointersToElements, helpers));
    
    gotProductIDs_.push_back(bh.id());
    gotViews_.push_back(newview);
    Handle<View<ELEMENT> > h(&*newview, bh.provenance());
    result.swap(h);
  }

}
#endif
