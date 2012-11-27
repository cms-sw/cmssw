#ifndef FWCore_Framework_Event_h
#define FWCore_Framework_Event_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     Event
//
/**\class Event Event.h FWCore/Framework/interface/Event.h

Description: This is the primary interface for accessing EDProducts
from a single collision and inserting new derived products.

For its usage, see "FWCore/Framework/interface/PrincipalGetAdapter.h"

*/
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"

#include "boost/shared_ptr.hpp"

#include <memory>
#include <string>
#include <set>
#include <typeinfo>
#include <vector>

namespace edm {

  class ConstBranchDescription;
  class TriggerResultsByName;
  class TriggerResults;
  class TriggerNames;

  class Event : public EventBase {
  public:
    Event(EventPrincipal& ep, ModuleDescription const& md);
    virtual ~Event();

    // AUX functions are defined in EventBase
    EventAuxiliary const& eventAuxiliary() const {return aux_;}

    LuminosityBlock const&
    getLuminosityBlock() const {
      return *luminosityBlock_;
    }

    Run const&
    getRun() const;

    RunNumber_t
    run() const {return id().run();}

    template<typename PROD>
    bool
    get(ProductID const& oid, Handle<PROD>& result) const;

    // Template member overload to deal with Views.
    template<typename ELEMENT>
    bool
    get(ProductID const& oid, Handle<View<ELEMENT> >& result) const ;

    EventSelectionIDVector const& eventSelectionIDs() const;

    ProcessHistoryID const& processHistoryID() const;

    ///Put a new product.
    template<typename PROD>
    OrphanHandle<PROD>
    put(std::auto_ptr<PROD> product) {return put<PROD>(product, std::string());}

    ///Put a new product with a 'product instance name'
    template<typename PROD>
    OrphanHandle<PROD>
    put(std::auto_ptr<PROD> product, std::string const& productInstanceName);

    ///Returns a RefProd to a product before that product has been placed into the Event.
    /// The RefProd (and any Ref's made from it) will no work properly until after the
    /// Event has been committed (which happens after leaving the EDProducer::produce method)
    template<typename PROD>
    RefProd<PROD>
    getRefBeforePut() {return getRefBeforePut<PROD>(std::string());}

    template<typename PROD>
    RefProd<PROD>
    getRefBeforePut(std::string const& productInstanceName);

    template<typename PROD>
    bool
    getByLabel(InputTag const& tag, Handle<PROD>& result) const;

    template<typename PROD>
    bool
    getByLabel(std::string const& label, Handle<PROD>& result) const;

    template<typename PROD>
    bool
    getByLabel(std::string const& label, std::string const& productInstanceName, Handle<PROD>& result) const;

    template<typename PROD>
    void
    getManyByType(std::vector<Handle<PROD> >& results) const;

    // Template member overload to deal with Views.
    template<typename ELEMENT>
    bool
    getByLabel(std::string const& label,
               Handle<View<ELEMENT> >& result) const;

    template<typename ELEMENT>
    bool
    getByLabel(std::string const& label,
               std::string const& productInstanceName,
               Handle<View<ELEMENT> >& result) const;

    template<typename ELEMENT>
    bool
    getByLabel(InputTag const& tag, Handle<View<ELEMENT> >& result) const;

    template<typename ELEMENT>
    void
    fillView_(BasicHandle& bh,
              Handle<View<ELEMENT> >& result) const;

    Provenance
    getProvenance(BranchID const& theID) const;

    Provenance
    getProvenance(ProductID const& theID) const;

    void
    getAllProvenance(std::vector<Provenance const*>& provenances) const;

    // Return true if this Event has been subjected to a process with
    // the given processName, and false otherwise.
    // If true is returned, then ps is filled with the ParameterSet
    // used to configure the identified process.
    bool
    getProcessParameterSet(std::string const& processName, ParameterSet& ps) const;

    virtual ProcessHistory const&
    processHistory() const;

    size_t size() const;

    virtual edm::TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults) const;
    virtual TriggerResultsByName triggerResultsByName(std::string const& process) const;

    typedef std::vector<std::pair<WrapperOwningHolder, ConstBranchDescription const*> > ProductPtrVec;

  private:
    EventPrincipal const&
    eventPrincipal() const;

    EventPrincipal&
    eventPrincipal();

    ProductID
    makeProductID(ConstBranchDescription const& desc) const;

    //override used by EventBase class
    virtual BasicHandle getByLabelImpl(std::type_info const& iWrapperType, std::type_info const& iProductType, InputTag const& iTag) const;

    // commit_() is called to complete the transaction represented by
    // this PrincipalGetAdapter. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ProducerSourceBase;
    friend class DaqSource;
    friend class InputSource;
    friend class RawInputSource;
    friend class EDFilter;
    friend class EDProducer;

    void commit_(std::vector<BranchID>* previousParentage= 0, ParentageID* previousParentageId = 0);
    void commit_aux(ProductPtrVec& products, bool record_parents, std::vector<BranchID>* previousParentage = 0, ParentageID* previousParentageId = 0);

    BasicHandle
    getByProductID_(ProductID const& oid) const;

    ProductPtrVec& putProducts() {return putProducts_;}
    ProductPtrVec const& putProducts() const {return putProducts_;}

    ProductPtrVec& putProductsWithoutParents() {return putProductsWithoutParents_;}
    ProductPtrVec const& putProductsWithoutParents() const {return putProductsWithoutParents_;}

    PrincipalGetAdapter provRecorder_;

    // putProducts_ and putProductsWithoutParents_ are the holding
    // pens for EDProducts inserted into this PrincipalGetAdapter. Pointers
    // in these collections own the products to which they point.
    //
    ProductPtrVec putProducts_;               // keep parentage info for these
    ProductPtrVec putProductsWithoutParents_; // ... but not for these

    EventAuxiliary const& aux_;
    boost::shared_ptr<LuminosityBlock const> const luminosityBlock_;

    // gotBranchIDs_ must be mutable because it records all 'gets',
    // which do not logically modify the PrincipalGetAdapter. gotBranchIDs_ is
    // merely a cache reflecting what has been retreived from the
    // Principal class.
    typedef std::set<BranchID> BranchIDSet;
    mutable BranchIDSet gotBranchIDs_;
    void addToGotBranchIDs(Provenance const& prov) const;

    // We own the retrieved Views, and have to destroy them.
    mutable std::vector<boost::shared_ptr<ViewBase> > gotViews_;
  };

  // The following functions objects are used by Event::put, under the
  // control of a metafunction if, to put the given pair into the
  // right collection.
  template<typename PROD>
  struct RecordInParentless {
    typedef Event::ProductPtrVec ptrvec_t;
    void do_it(ptrvec_t& /*ignored*/,
               ptrvec_t& used,
               WrapperOwningHolder const& edp,
               ConstBranchDescription const* desc) const {
      used.push_back(std::make_pair(edp, desc));
    }
  };

  template<typename PROD>
  struct RecordInParentfull {
    typedef Event::ProductPtrVec ptrvec_t;

    void do_it(ptrvec_t& used,
               ptrvec_t& /*ignored*/,
               WrapperOwningHolder const& edp,
               ConstBranchDescription const* desc) const {
      used.push_back(std::make_pair(edp, desc));
    }
  };


  template<typename PROD>
  bool
  Event::get(ProductID const& oid, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = this->getByProductID_(oid);
    convert_handle(bh, result);  // throws on conversion error
    if(bh.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*bh.provenance());
    return true;
  }

  template<typename ELEMENT>
  bool
  Event::get(ProductID const& oid, Handle<View<ELEMENT> >& result) const {
      result.clear();
      BasicHandle bh = this->getByProductID_(oid);

      if(bh.failedToGet()) {
          boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound));
          *whyFailed
              << "get View by ID failed: no product with ID = " << oid <<"\n";
          Handle<View<ELEMENT> > temp(whyFailed);
          result.swap(temp);
          return false;
      }

      fillView_(bh, result);
      return true;
  }

  template<typename PROD>
  OrphanHandle<PROD>
  Event::put(std::auto_ptr<PROD> product, std::string const& productInstanceName) {
    if(product.get() == 0) {                // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Event", typeID, productInstanceName);
    }

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    typename boost::mpl::if_c<detail::has_postinsert<PROD>::value,
      DoPostInsert<PROD>,
      DoNotPostInsert<PROD> >::type maybe_inserter;
    maybe_inserter(product.get());

    ConstBranchDescription const& desc =
      provRecorder_.getBranchDescription(TypeID(*product), productInstanceName);

    WrapperOwningHolder edp(new Wrapper<PROD>(product), Wrapper<PROD>::getInterface());

    typename boost::mpl::if_c<detail::has_donotrecordparents<PROD>::value,
      RecordInParentless<PROD>,
      RecordInParentfull<PROD> >::type parentage_recorder;
    parentage_recorder.do_it(putProducts(),
                             putProductsWithoutParents(),
                             edp,
                             &desc);

    //  putProducts().push_back(std::make_pair(edp, &desc));

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.

    return(OrphanHandle<PROD>(static_cast<Wrapper<PROD> const*>(edp.wrapper())->product(), makeProductID(desc)));
  }

  template<typename PROD>
  RefProd<PROD>
  Event::getRefBeforePut(std::string const& productInstanceName) {
    PROD* p = 0;
    ConstBranchDescription const& desc =
      provRecorder_.getBranchDescription(TypeID(*p), productInstanceName);

    //should keep track of what Ref's have been requested and make sure they are 'put'
    return RefProd<PROD>(makeProductID(desc), provRecorder_.prodGetter());
  }

  template<typename PROD>
  bool
  Event::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    bool ok = provRecorder_.getByLabel(tag, result);
    if(ok) {
      addToGotBranchIDs(*result.provenance());
    }
    return ok;
  }

  template<typename PROD>
  bool
  Event::getByLabel(std::string const& label, Handle<PROD>& result) const {
    bool ok = provRecorder_.getByLabel(label, result);
    if(ok) {
      addToGotBranchIDs(*result.provenance());
    }
    return ok;
  }

  template<typename PROD>
  bool
  Event::getByLabel(std::string const& label,
                    std::string const& productInstanceName,
                    Handle<PROD>& result) const {
    bool ok = provRecorder_.getByLabel(label, productInstanceName, result);
    if(ok) {
      addToGotBranchIDs(*result.provenance());
    }
    return ok;
  }

  template<typename PROD>
  void
  Event::getManyByType(std::vector<Handle<PROD> >& results) const {
    provRecorder_.getManyByType(results);
    for(typename std::vector<Handle<PROD> >::const_iterator it = results.begin(), itEnd = results.end();
        it != itEnd; ++it) {
      addToGotBranchIDs(*it->provenance());
    }
  }

  template<typename ELEMENT>
  bool
  Event::getByLabel(std::string const& moduleLabel, Handle<View<ELEMENT> >& result) const {
    return getByLabel(moduleLabel, std::string(), result);
  }

  template<typename ELEMENT>
  bool
  Event::getByLabel(std::string const& moduleLabel,
                    std::string const& productInstanceName,
                    Handle<View<ELEMENT> >& result) const {
    result.clear();

    TypeID typeID(typeid(ELEMENT));

    BasicHandle bh;
    std::string processName; // empty
    int nFound = provRecorder_.getMatchingSequenceByLabel_(typeID,
                                                           moduleLabel,
                                                           productInstanceName,
                                                           processName,
                                                           bh);

    if(nFound == 0) {
      boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound));
      *whyFailed
        << "getByLabel: Found zero products matching all criteria\n"
        << "Looking for sequence of type: " << typeID << "\n"
        << "Looking for module label: " << moduleLabel << "\n"
        << "Looking for productInstanceName: " << productInstanceName << "\n";
      Handle<View<ELEMENT> > temp(whyFailed);
      result.swap(temp);
      return false;
    }
    if(nFound > 1) {
      Exception e(errors::ProductNotFound);
      e << "getByLabel: Found more than one product matching all criteria\n"
        << "Looking for sequence of type: " << typeID << "\n"
        << "Looking for module label: " << moduleLabel << "\n"
        << "Looking for productInstanceName: " << productInstanceName << "\n";
      e.raise();
    }

    fillView_(bh, result);
    return true;
  }

  template<typename ELEMENT>
    bool
    Event::getByLabel(InputTag const& tag, Handle<View<ELEMENT> >& result) const {
    result.clear();
    if(tag.process().empty()) {
      return getByLabel(tag.label(), tag.instance(), result);
    } else {
      TypeID typeID(typeid(ELEMENT));

      BasicHandle bh;
      int nFound = provRecorder_.getMatchingSequenceByLabel_(typeID,
                                                             tag.label(),
                                                             tag.instance(),
                                                             tag.process(),
                                                             bh);

      if(nFound == 0) {
        boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound));
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
      if(nFound > 1) {
        Exception e (errors::ProductNotFound);
        e << "getByLabel: Found more than one product matching all criteria\n"
          << "Looking for sequence of type: " << typeID << "\n"
          << "Looking for module label: " << tag.label() << "\n"
          << "Looking for productInstanceName: " << tag.instance() << "\n"
          << "Looking for processName: "<<tag.process() <<"\n";
        e.raise();
      }

      fillView_(bh, result);
      return true;
    }
    return false;
  }

  template<typename ELEMENT>
  void
  Event::fillView_(BasicHandle& bh, Handle<View<ELEMENT> >& result) const {
    std::vector<void const*> pointersToElements;
    // the following is a shared pointer.
    // It is not initialized here
    helper_vector_ptr helpers;
    // the following must initialize the
    //  shared pointer and fill the helper vector
    bh.interface()->fillView(bh.wrapper(), bh.id(), pointersToElements, helpers);

    boost::shared_ptr<View<ELEMENT> >
      newview(new View<ELEMENT>(pointersToElements, helpers));

    addToGotBranchIDs(*bh.provenance());
    gotViews_.push_back(newview);
    Handle<View<ELEMENT> > h(&*newview, bh.provenance());
    result.swap(h);
  }
}
#endif
