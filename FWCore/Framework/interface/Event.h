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
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/FillViewHelperVector.h"
#include "DataFormats/Common/interface/FunctorHandleExceptionFactory.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>
#include <string>
#include <set>
#include <typeinfo>
#include <type_traits>
#include <vector>

class testEventGetRefBeforePut;

namespace edm {

  class BranchDescription;
  class ModuleCallingContext;
  class TriggerResultsByName;
  class TriggerResults;
  class TriggerNames;
  class EDConsumerBase;
  class EDProductGetter;
  class ProducerBase;
  class SharedResourcesAcquirer;
  namespace stream {
    template< typename T> class ProducingModuleAdaptorBase;
  }

  class Event : public EventBase {
  public:
    Event(EventPrincipal const& ep, ModuleDescription const& md,
          ModuleCallingContext const*);
    ~Event() override;

    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer);

    void setSharedResourcesAcquirer( SharedResourcesAcquirer* iResourceAcquirer);

    // AUX functions are defined in EventBase
    EventAuxiliary const& eventAuxiliary() const override {return aux_;}

    ///\return The id for the particular Stream processing the Event
    StreamID streamID() const {
      return streamID_;
    }

    LuminosityBlock const&
    getLuminosityBlock() const {
      return *luminosityBlock_;
    }

    Run const&
    getRun() const;

    RunNumber_t
    run() const {return id().run();}

    /**If you are caching data from the Event, you should also keep
     this number.  If this number changes then you know that
     the data you have cached is invalid.
     The value of '0' will never be returned so you can use that to
     denote that you have not yet checked the value.
     */
    typedef unsigned long CacheIdentifier_t;
    CacheIdentifier_t
    cacheIdentifier() const;

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
    put(std::unique_ptr<PROD> product) {return put<PROD>(std::move(product), std::string());}

    ///Put a new product with a 'product instance name'
    template<typename PROD>
    OrphanHandle<PROD>
    put(std::unique_ptr<PROD> product, std::string const& productInstanceName);

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

    template<typename PROD>
    bool
    getByToken(EDGetToken token, Handle<PROD>& result) const;

    template<typename PROD>
    bool
    getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const;

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
    bool
    getByToken(EDGetToken token, Handle<View<ELEMENT>>& result) const;

    template<typename ELEMENT>
    bool
    getByToken(EDGetTokenT<View<ELEMENT>> token, Handle<View<ELEMENT>>& result) const;


    template<typename ELEMENT>
    void
    fillView_(BasicHandle& bh,
              Handle<View<ELEMENT> >& result) const;

    Provenance
    getProvenance(BranchID const& theID) const;

    Provenance
    getProvenance(ProductID const& theID) const;

    // Get the provenance for all products that may be in the event
    void
    getAllProvenance(std::vector<Provenance const*>& provenances) const;

    // Get the provenance for all products that may be in the event,
    // excluding the per-event provenance (the parentage information).
    // The excluded information may change from event to event.
    void
    getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const;

    // Return true if this Event has been subjected to a process with
    // the given processName, and false otherwise.
    // If true is returned, then ps is filled with the ParameterSet
    // used to configure the identified process.
    bool
    getProcessParameterSet(std::string const& processName, ParameterSet& ps) const;

    ProcessHistory const&
    processHistory() const override;

    edm::ParameterSet const*
    parameterSet(edm::ParameterSetID const& psID) const override;

    size_t size() const;

    edm::TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults) const override;
    TriggerResultsByName triggerResultsByName(edm::TriggerResults const& triggerResults) const override;

    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }

    void labelsForToken(EDGetToken const& iToken, ProductLabels& oLabels) const { provRecorder_.labelsForToken(iToken, oLabels); }

    typedef std::vector<std::pair<edm::propagate_const<std::unique_ptr<WrapperBase>>, BranchDescription const*> > ProductPtrVec;

    EDProductGetter const&
    productGetter() const;

  private:
    //for testing
    friend class ::testEventGetRefBeforePut;

    EventPrincipal const&
    eventPrincipal() const;

    ProductID
    makeProductID(BranchDescription const& desc) const;

    //override used by EventBase class
    BasicHandle getByLabelImpl(std::type_info const& iWrapperType, std::type_info const& iProductType, InputTag const& iTag) const override;

    //override used by EventBase class
    BasicHandle getImpl(std::type_info const& iProductType, ProductID const& pid) const override;

    // commit_() is called to complete the transaction represented by
    // this PrincipalGetAdapter. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ProducerSourceBase;
    friend class InputSource;
    friend class RawInputSource;
    friend class ProducerBase;
    template<typename T> friend class stream::ProducingModuleAdaptorBase;

    void commit_(std::vector<edm::ProductResolverIndex> const& iShouldPut, std::vector<BranchID>* previousParentage= nullptr, ParentageID* previousParentageId = nullptr);
    void commit_aux(ProductPtrVec& products, bool record_parents, std::vector<BranchID>* previousParentage = nullptr, ParentageID* previousParentageId = nullptr);

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
    std::shared_ptr<LuminosityBlock const> const luminosityBlock_;

    // gotBranchIDs_ must be mutable because it records all 'gets',
    // which do not logically modify the PrincipalGetAdapter. gotBranchIDs_ is
    // merely a cache reflecting what has been retreived from the
    // Principal class.
    typedef std::set<BranchID> BranchIDSet;
    mutable BranchIDSet gotBranchIDs_;
    void addToGotBranchIDs(Provenance const& prov) const;

    // We own the retrieved Views, and have to destroy them.
    mutable std::vector<std::shared_ptr<ViewBase> > gotViews_;

    StreamID streamID_;
    ModuleCallingContext const* moduleCallingContext_;

    static const std::string emptyString_;
  };

  // The following functions objects are used by Event::put, under the
  // control of a metafunction if, to put the given pair into the
  // right collection.
  template<typename PROD>
  struct RecordInParentless {
    typedef Event::ProductPtrVec ptrvec_t;
    void do_it(ptrvec_t& /*ignored*/,
               ptrvec_t& used,
               std::unique_ptr<Wrapper<PROD> > wp,
               BranchDescription const* desc) const {
      used.emplace_back(std::move(wp), desc);
    }
  };

  template<typename PROD>
  struct RecordInParentfull {
    typedef Event::ProductPtrVec ptrvec_t;

    void do_it(ptrvec_t& used,
               ptrvec_t& /*ignored*/,
               std::unique_ptr<Wrapper<PROD> > wp,
               BranchDescription const* desc) const {
      used.emplace_back(std::move(wp), desc);
    }
  };


  template<typename PROD>
  bool
  Event::get(ProductID const& oid, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = this->getByProductID_(oid);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if(result.failedToGet()) {
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
          Handle<View<ELEMENT> > temp(makeHandleExceptionFactory([oid]()->std::shared_ptr<cms::Exception> {
            std::shared_ptr<cms::Exception> whyFailed = std::make_shared<edm::Exception>(edm::errors::ProductNotFound);
            *whyFailed
            << "get View by ID failed: no product with ID = " << oid <<"\n";
            return whyFailed;
          }));
          result.swap(temp);
          return false;
      }

      fillView_(bh, result);
      return true;
  }

  template<typename PROD>
  OrphanHandle<PROD>
  Event::put(std::unique_ptr<PROD> product, std::string const& productInstanceName) {
    if(product.get() == 0) {                // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Event", typeID, productInstanceName);
    }

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    std::conditional_t<detail::has_postinsert<PROD>::value,
                       DoPostInsert<PROD>,
                       DoNotPostInsert<PROD>> maybe_inserter;
    maybe_inserter(product.get());

    BranchDescription const& desc =
      provRecorder_.getBranchDescription(TypeID(*product), productInstanceName);

    std::unique_ptr<Wrapper<PROD> > wp(new Wrapper<PROD>(std::move(product)));
    PROD const* prod = wp->product();

    std::conditional_t<detail::has_donotrecordparents<PROD>::value,
                       RecordInParentless<PROD>,
                       RecordInParentfull<PROD>> parentage_recorder;
    parentage_recorder.do_it(putProducts(),
                             putProductsWithoutParents(),
                             std::move(wp),
                             &desc);

    //  putProducts().push_back(std::make_pair(edp, &desc));

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.

    return(OrphanHandle<PROD>(prod, makeProductID(desc)));
  }

  template<typename PROD>
  RefProd<PROD>
  Event::getRefBeforePut(std::string const& productInstanceName) {
    PROD* p = nullptr;
    BranchDescription const& desc =
      provRecorder_.getBranchDescription(TypeID(*p), productInstanceName);

    //should keep track of what Ref's have been requested and make sure they are 'put'
    return RefProd<PROD>(makeProductID(desc), provRecorder_.prodGetter());
  }

  template<typename PROD>
  bool
  Event::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(typeid(PROD)), tag, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  template<typename PROD>
  bool
  Event::getByLabel(std::string const& label,
                    std::string const& productInstanceName,
                    Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(typeid(PROD)), label, productInstanceName, emptyString_, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  template<typename PROD>
  bool
  Event::getByLabel(std::string const& label, Handle<PROD>& result) const {
    return getByLabel(label, emptyString_, result);
  }

  template<typename PROD>
  void
  Event::getManyByType(std::vector<Handle<PROD> >& results) const {
    provRecorder_.getManyByType(results, moduleCallingContext_);
    for(typename std::vector<Handle<PROD> >::const_iterator it = results.begin(), itEnd = results.end();
        it != itEnd; ++it) {
      addToGotBranchIDs(*it->provenance());
    }
  }

  template<typename PROD>
  bool
  Event::getByToken(EDGetToken token, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)),PRODUCT_TYPE, token, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  template<typename PROD>
  bool
  Event::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)),PRODUCT_TYPE, token, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }


  template<typename ELEMENT>
  bool
  Event::getByLabel(InputTag const& tag, Handle<View<ELEMENT> >& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getMatchingSequenceByLabel_(TypeID(typeid(ELEMENT)), tag, moduleCallingContext_);
    if(bh.failedToGet()) {
      Handle<View<ELEMENT> > h(std::move(bh.whyFailedFactory()));
      h.swap(result);
      return false;
    }
    fillView_(bh, result);
    return true;
  }

  template<typename ELEMENT>
  bool
  Event::getByLabel(std::string const& moduleLabel,
                    std::string const& productInstanceName,
                    Handle<View<ELEMENT> >& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getMatchingSequenceByLabel_(TypeID(typeid(ELEMENT)), moduleLabel, productInstanceName, emptyString_, moduleCallingContext_);
    if(bh.failedToGet()) {
      Handle<View<ELEMENT> > h(std::move(bh.whyFailedFactory()));
      h.swap(result);
      return false;
    }
    fillView_(bh, result);
    return true;
  }

  template<typename ELEMENT>
  bool
  Event::getByLabel(std::string const& moduleLabel, Handle<View<ELEMENT> >& result) const {
    return getByLabel(moduleLabel, emptyString_, result);
  }

  template<typename ELEMENT>
  bool
  Event::getByToken(EDGetToken token, Handle<View<ELEMENT>>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(ELEMENT)),ELEMENT_TYPE, token, moduleCallingContext_);
    if(bh.failedToGet()) {
      Handle<View<ELEMENT> > h(std::move(bh.whyFailedFactory()));
      h.swap(result);
      return false;
    }
    fillView_(bh, result);
    return true;
  }

  template<typename ELEMENT>
  bool
  Event::getByToken(EDGetTokenT<View<ELEMENT>> token, Handle<View<ELEMENT>>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(ELEMENT)),ELEMENT_TYPE, token, moduleCallingContext_);
    if(bh.failedToGet()) {
      Handle<View<ELEMENT> > h(std::move(bh.whyFailedFactory()));
      h.swap(result);
      return false;
    }
    fillView_(bh, result);
    return true;
  }


  template<typename ELEMENT>
  void
  Event::fillView_(BasicHandle& bh, Handle<View<ELEMENT> >& result) const {
    std::vector<void const*> pointersToElements;
    FillViewHelperVector helpers;
    // the following must initialize the
    //  fill the helper vector
    bh.wrapper()->fillView(bh.id(), pointersToElements, helpers);

    auto newview = std::make_shared<View<ELEMENT> >(pointersToElements, helpers, &(productGetter()));

    addToGotBranchIDs(*bh.provenance());
    gotViews_.push_back(newview);
    Handle<View<ELEMENT> > h(&*newview, bh.provenance());
    result.swap(h);
  }
}
#endif
