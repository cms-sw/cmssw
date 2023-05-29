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
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <typeinfo>
#include <type_traits>
#include <vector>

class testEventGetRefBeforePut;
class testEvent;

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
    template <typename T>
    class ProducingModuleAdaptorBase;
  }

  class Event : public EventBase {
  public:
    Event(EventTransitionInfo const&, ModuleDescription const&, ModuleCallingContext const*);
    Event(EventPrincipal const&, ModuleDescription const&, ModuleCallingContext const*);
    ~Event() override;

    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer);

    void setSharedResourcesAcquirer(SharedResourcesAcquirer* iResourceAcquirer);

    void setProducerCommon(ProducerBase const* iProd, std::vector<BranchID>* previousParentage);

    void setProducer(ProducerBase const* iProd,
                     std::vector<BranchID>* previousParentage,
                     std::vector<BranchID>* gotBranchIDsFromAcquire = nullptr);

    void setProducerForAcquire(ProducerBase const* iProd,
                               std::vector<BranchID>* previousParentage,
                               std::vector<BranchID>& gotBranchIDsFromAcquire);

    // AUX functions are defined in EventBase
    EventAuxiliary const& eventAuxiliary() const override { return aux_; }

    ///\return The id for the particular Stream processing the Event
    StreamID streamID() const { return streamID_; }

    LuminosityBlock const& getLuminosityBlock() const {
      if (not luminosityBlock_) {
        fillLuminosityBlock();
      }
      return *luminosityBlock_;
    }

    Run const& getRun() const;

    RunNumber_t run() const { return id().run(); }

    /**If you are caching data from the Event, you should also keep
     this number.  If this number changes then you know that
     the data you have cached is invalid.
     The value of '0' will never be returned so you can use that to
     denote that you have not yet checked the value.
     */
    typedef unsigned long CacheIdentifier_t;
    CacheIdentifier_t cacheIdentifier() const;

    template <typename PROD>
    bool get(ProductID const& oid, Handle<PROD>& result) const;

    // Template member overload to deal with Views.
    template <typename ELEMENT>
    bool get(ProductID const& oid, Handle<View<ELEMENT>>& result) const;

    EventSelectionIDVector const& eventSelectionIDs() const;

    ProcessHistoryID const& processHistoryID() const;

    ///Put a new product.
    template <typename PROD>
    OrphanHandle<PROD> put(std::unique_ptr<PROD> product) {
      return put<PROD>(std::move(product), std::string());
    }

    ///Put a new product with a 'product instance name'
    template <typename PROD>
    OrphanHandle<PROD> put(std::unique_ptr<PROD> product, std::string const& productInstanceName);

    template <typename PROD>
    OrphanHandle<PROD> put(EDPutToken token, std::unique_ptr<PROD> product);

    template <typename PROD>
    OrphanHandle<PROD> put(EDPutTokenT<PROD> token, std::unique_ptr<PROD> product);

    ///puts a new product
    template <typename PROD, typename... Args>
    OrphanHandle<PROD> emplace(EDPutTokenT<PROD> token, Args&&... args);

    template <typename PROD, typename... Args>
    OrphanHandle<PROD> emplace(EDPutToken token, Args&&... args);

    ///Returns a RefProd to a product before that product has been placed into the Event.
    /// The RefProd (and any Ref's made from it) will no work properly until after the
    /// Event has been committed (which happens after leaving the EDProducer::produce method)
    template <typename PROD>
    RefProd<PROD> getRefBeforePut() {
      return getRefBeforePut<PROD>(std::string());
    }

    template <typename PROD>
    RefProd<PROD> getRefBeforePut(std::string const& productInstanceName);

    template <typename PROD>
    RefProd<PROD> getRefBeforePut(EDPutTokenT<PROD>);

    template <typename PROD>
    RefProd<PROD> getRefBeforePut(EDPutToken);

    template <typename PROD>
    bool getByLabel(InputTag const& tag, Handle<PROD>& result) const;

    template <typename PROD>
    bool getByLabel(std::string const& label, Handle<PROD>& result) const;

    template <typename PROD>
    bool getByLabel(std::string const& label, std::string const& productInstanceName, Handle<PROD>& result) const;

    template <typename PROD>
    void getManyByType(std::vector<Handle<PROD>>& results) const;

    template <typename PROD>
    bool getByToken(EDGetToken token, Handle<PROD>& result) const;

    template <typename PROD>
    bool getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const;

    template <typename PROD>
    Handle<PROD> getHandle(EDGetTokenT<PROD> token) const;

    template <typename PROD>
    PROD const& get(EDGetTokenT<PROD> token) const noexcept(false);

    // Template member overload to deal with Views.
    template <typename ELEMENT>
    bool getByLabel(std::string const& label, Handle<View<ELEMENT>>& result) const;

    template <typename ELEMENT>
    bool getByLabel(std::string const& label,
                    std::string const& productInstanceName,
                    Handle<View<ELEMENT>>& result) const;

    template <typename ELEMENT>
    bool getByLabel(InputTag const& tag, Handle<View<ELEMENT>>& result) const;

    template <typename ELEMENT>
    bool getByToken(EDGetToken token, Handle<View<ELEMENT>>& result) const;

    template <typename ELEMENT>
    bool getByToken(EDGetTokenT<View<ELEMENT>> token, Handle<View<ELEMENT>>& result) const;

    template <typename ELEMENT>
    Handle<View<ELEMENT>> getHandle(EDGetTokenT<View<ELEMENT>> token) const;

    template <typename ELEMENT>
    View<ELEMENT> const& get(EDGetTokenT<View<ELEMENT>> token) const noexcept(false);

    template <typename ELEMENT>
    Handle<View<ELEMENT>> fillView_(BasicHandle& bh) const;

    Provenance const& getProvenance(BranchID const& theID) const;

    Provenance const& getProvenance(ProductID const& theID) const;

    StableProvenance const& getStableProvenance(BranchID const& theID) const;

    StableProvenance const& getStableProvenance(ProductID const& theID) const;

    // Get the provenance for all products that may be in the event
    void getAllProvenance(std::vector<Provenance const*>& provenances) const;

    // Get the provenance for all products that may be in the event,
    // excluding the per-event provenance (the parentage information).
    // The excluded information may change from event to event.
    void getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const;

    // Return true if this Event has been subjected to a process with
    // the given processName, and false otherwise.
    // If true is returned, then ps is filled with the ParameterSet
    // used to configure the identified process.
    bool getProcessParameterSet(std::string const& processName, ParameterSet& ps) const;

    ProcessHistory const& processHistory() const override;

    edm::ParameterSet const* parameterSet(edm::ParameterSetID const& psID) const override;

    size_t size() const;

    edm::TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults) const override;
    TriggerResultsByName triggerResultsByName(edm::TriggerResults const& triggerResults) const override;

    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }

    void labelsForToken(EDGetToken const& iToken, ProductLabels& oLabels) const {
      provRecorder_.labelsForToken(iToken, oLabels);
    }

    typedef std::vector<edm::propagate_const<std::unique_ptr<WrapperBase>>> ProductPtrVec;

    EDProductGetter const& productGetter() const;

    unsigned int processBlockIndex(std::string const& processName) const {
      return provRecorder_.processBlockIndex(processName);
    }

  private:
    //for testing
    friend class ::testEventGetRefBeforePut;
    friend class ::testEvent;

    EventPrincipal const& eventPrincipal() const;

    void fillLuminosityBlock() const;

    ProductID makeProductID(BranchDescription const& desc) const;

    //override used by EventBase class
    BasicHandle getByLabelImpl(std::type_info const& iWrapperType,
                               std::type_info const& iProductType,
                               InputTag const& iTag) const override;
    BasicHandle getByTokenImpl(std::type_info const& iProductType, EDGetToken iToken) const override;

    //override used by EventBase class
    BasicHandle getImpl(std::type_info const& iProductType, ProductID const& pid) const override;

    template <typename PROD>
    OrphanHandle<PROD> putImpl(EDPutToken::value_type token, std::unique_ptr<PROD> product);

    template <typename PROD, typename... Args>
    OrphanHandle<PROD> emplaceImpl(EDPutToken::value_type token, Args&&... args);

    // commit_() is called to complete the transaction represented by
    // this PrincipalGetAdapter. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ProducerSourceBase;
    friend class InputSource;
    friend class RawInputSource;
    friend class ProducerBase;
    template <typename T>
    friend class stream::ProducingModuleAdaptorBase;

    void commit_(std::vector<edm::ProductResolverIndex> const& iShouldPut, ParentageID* previousParentageId = nullptr);
    void commit_aux(ProductPtrVec& products, ParentageID* previousParentageId = nullptr);

    BasicHandle getByProductID_(ProductID const& oid) const;

    ProductPtrVec& putProducts() { return putProducts_; }
    ProductPtrVec const& putProducts() const { return putProducts_; }

    PrincipalGetAdapter provRecorder_;

    // putProducts_ is a holding pen for EDProducts inserted into this
    // PrincipalGetAdapter.
    //
    ProductPtrVec putProducts_;

    EventAuxiliary const& aux_;
    // measurable performance gain by only creating LuminosityBlock when needed
    // mutables are allowed in this case because edm::Event is only accessed by one thread
    CMS_SA_ALLOW mutable std::optional<LuminosityBlock> luminosityBlock_;

    // gotBranchIDs_ must be mutable because it records all 'gets',
    // which do not logically modify the PrincipalGetAdapter. gotBranchIDs_ is
    // merely a cache reflecting what has been retrieved from the
    // Principal class.
    typedef std::unordered_set<BranchID::value_type> BranchIDSet;
    CMS_SA_ALLOW mutable BranchIDSet gotBranchIDs_;
    CMS_SA_ALLOW mutable std::vector<bool> gotBranchIDsFromPrevious_;
    std::vector<BranchID>* previousBranchIDs_ = nullptr;
    std::vector<BranchID>* gotBranchIDsFromAcquire_ = nullptr;

    void addToGotBranchIDs(Provenance const& prov) const;
    void addToGotBranchIDs(BranchID const& branchID) const;

    // We own the retrieved Views, and have to destroy them.
    CMS_SA_ALLOW mutable std::vector<std::shared_ptr<ViewBase>> gotViews_;

    StreamID streamID_;
    ModuleCallingContext const* moduleCallingContext_;

    static const std::string emptyString_;
  };

  template <typename PROD>
  bool Event::get(ProductID const& oid, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = this->getByProductID_(oid);
    result = convert_handle_check_type<PROD>(std::move(bh));  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*bh.provenance());
    return true;
  }

  template <typename ELEMENT>
  bool Event::get(ProductID const& oid, Handle<View<ELEMENT>>& result) const {
    result.clear();
    BasicHandle bh = this->getByProductID_(oid);

    if (bh.failedToGet()) {
      result = Handle<View<ELEMENT>>(makeHandleExceptionFactory([oid]() -> std::shared_ptr<cms::Exception> {
        std::shared_ptr<cms::Exception> whyFailed = std::make_shared<edm::Exception>(edm::errors::ProductNotFound);
        *whyFailed << "get View by ID failed: no product with ID = " << oid << "\n";
        return whyFailed;
      }));
      return false;
    }

    result = fillView_<ELEMENT>(bh);
    return true;
  }

  template <typename PROD>
  OrphanHandle<PROD> Event::putImpl(EDPutToken::value_type index, std::unique_ptr<PROD> product) {
    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    std::conditional_t<detail::has_postinsert<PROD>::value, DoPostInsert<PROD>, DoNotPostInsert<PROD>> maybe_inserter;
    maybe_inserter(product.get());

    assert(index < putProducts().size());

    std::unique_ptr<Wrapper<PROD>> wp(new Wrapper<PROD>(std::move(product)));
    PROD const* prod = wp->product();

    putProducts()[index] = std::move(wp);
    auto const& prodID = provRecorder_.getProductID(index);
    return (OrphanHandle<PROD>(prod, prodID));
  }

  template <typename PROD>
  OrphanHandle<PROD> Event::put(std::unique_ptr<PROD> product, std::string const& productInstanceName) {
    if (UNLIKELY(product.get() == nullptr)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Event", typeID, productInstanceName);
    }

    auto index = provRecorder_.getPutTokenIndex(TypeID(*product), productInstanceName);
    return putImpl(index, std::move(product));
  }

  template <typename PROD>
  OrphanHandle<PROD> Event::put(EDPutTokenT<PROD> token, std::unique_ptr<PROD> product) {
    if (UNLIKELY(product.get() == nullptr)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Event", typeID, provRecorder_.productInstanceLabel(token));
    }
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Event", typeid(PROD));
    }
    return putImpl(token.index(), std::move(product));
  }

  template <typename PROD>
  OrphanHandle<PROD> Event::put(EDPutToken token, std::unique_ptr<PROD> product) {
    if (UNLIKELY(product.get() == nullptr)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Event", typeID, provRecorder_.productInstanceLabel(token));
    }
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Event", typeid(PROD));
    }
    if (UNLIKELY(provRecorder_.getTypeIDForPutTokenIndex(token.index()) != TypeID{typeid(PROD)})) {
      principal_get_adapter_detail::throwOnPutOfWrongType(typeid(PROD),
                                                          provRecorder_.getTypeIDForPutTokenIndex(token.index()));
    }

    return putImpl(token.index(), std::move(product));
  }

  template <typename PROD, typename... Args>
  OrphanHandle<PROD> Event::emplace(EDPutTokenT<PROD> token, Args&&... args) {
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Event", typeid(PROD));
    }
    return emplaceImpl<PROD>(token.index(), std::forward<Args>(args)...);
  }

  template <typename PROD, typename... Args>
  OrphanHandle<PROD> Event::emplace(EDPutToken token, Args&&... args) {
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Event", typeid(PROD));
    }
    if (UNLIKELY(provRecorder_.getTypeIDForPutTokenIndex(token.index()) != TypeID{typeid(PROD)})) {
      principal_get_adapter_detail::throwOnPutOfWrongType(typeid(PROD),
                                                          provRecorder_.getTypeIDForPutTokenIndex(token.index()));
    }

    return emplaceImpl(token.index(), std::forward<Args>(args)...);
  }

  template <typename PROD, typename... Args>
  OrphanHandle<PROD> Event::emplaceImpl(EDPutToken::value_type index, Args&&... args) {
    assert(index < putProducts().size());

    std::unique_ptr<Wrapper<PROD>> wp(new Wrapper<PROD>(WrapperBase::Emplace{}, std::forward<Args>(args)...));

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    std::conditional_t<detail::has_postinsert<PROD>::value, DoPostInsert<PROD>, DoNotPostInsert<PROD>> maybe_inserter;
    maybe_inserter(&(wp->bareProduct()));

    PROD const* prod = wp->product();

    putProducts()[index] = std::move(wp);
    auto const& prodID = provRecorder_.getProductID(index);
    return (OrphanHandle<PROD>(prod, prodID));
  }

  template <typename PROD>
  RefProd<PROD> Event::getRefBeforePut(std::string const& productInstanceName) {
    auto index = provRecorder_.getPutTokenIndex(TypeID{typeid(PROD)}, productInstanceName);

    //should keep track of what Ref's have been requested and make sure they are 'put'
    return RefProd<PROD>(provRecorder_.getProductID(index), provRecorder_.prodGetter());
  }

  template <typename PROD>
  RefProd<PROD> Event::getRefBeforePut(EDPutTokenT<PROD> token) {
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Event", typeid(PROD));
    }
    return RefProd<PROD>(provRecorder_.getProductID(token.index()), provRecorder_.prodGetter());
  }

  template <typename PROD>
  RefProd<PROD> Event::getRefBeforePut(EDPutToken token) {
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Event", typeid(PROD));
    }
    if (UNLIKELY(provRecorder_.getTypeIDForPutTokenIndex(token.index()) != TypeID{typeid(PROD)})) {
      principal_get_adapter_detail::throwOnPutOfWrongType(typeid(PROD),
                                                          provRecorder_.getTypeIDForPutTokenIndex(token.index()));
    }
    return RefProd<PROD>(provRecorder_.getProductID(token.index()), provRecorder_.prodGetter());
  }

  template <typename PROD>
  bool Event::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(typeid(PROD)), tag, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if UNLIKELY (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  template <typename PROD>
  bool Event::getByLabel(std::string const& label, std::string const& productInstanceName, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByLabel_(
        TypeID(typeid(PROD)), label, productInstanceName, emptyString_, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if UNLIKELY (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  template <typename PROD>
  bool Event::getByLabel(std::string const& label, Handle<PROD>& result) const {
    return getByLabel(label, emptyString_, result);
  }

  template <typename PROD>
  void Event::getManyByType(std::vector<Handle<PROD>>& results) const {
    provRecorder_.getManyByType(results, moduleCallingContext_);
    for (typename std::vector<Handle<PROD>>::const_iterator it = results.begin(), itEnd = results.end(); it != itEnd;
         ++it) {
      addToGotBranchIDs(*it->provenance());
    }
  }

  template <typename PROD>
  bool Event::getByToken(EDGetToken token, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if UNLIKELY (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  template <typename PROD>
  bool Event::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));
    if UNLIKELY (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  template <typename PROD>
  Handle<PROD> Event::getHandle(EDGetTokenT<PROD> token) const {
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    auto result = convert_handle<PROD>(std::move(bh));
    if LIKELY (not result.failedToGet()) {
      addToGotBranchIDs(*result.provenance());
    }
    return result;
  }

  template <typename PROD>
  PROD const& Event::get(EDGetTokenT<PROD> token) const noexcept(false) {
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    auto result = convert_handle<PROD>(std::move(bh));
    if LIKELY (not result.failedToGet()) {
      addToGotBranchIDs(*result.provenance());
    }
    return *result;
  }

  template <typename ELEMENT>
  bool Event::getByLabel(InputTag const& tag, Handle<View<ELEMENT>>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getMatchingSequenceByLabel_(TypeID(typeid(ELEMENT)), tag, moduleCallingContext_);
    if UNLIKELY (bh.failedToGet()) {
      Handle<View<ELEMENT>> h(std::move(bh.whyFailedFactory()));
      h.swap(result);
      return false;
    }
    result = fillView_<ELEMENT>(bh);
    return true;
  }

  template <typename ELEMENT>
  bool Event::getByLabel(std::string const& moduleLabel,
                         std::string const& productInstanceName,
                         Handle<View<ELEMENT>>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getMatchingSequenceByLabel_(
        TypeID(typeid(ELEMENT)), moduleLabel, productInstanceName, emptyString_, moduleCallingContext_);
    if UNLIKELY (bh.failedToGet()) {
      Handle<View<ELEMENT>> h(std::move(bh.whyFailedFactory()));
      h.swap(result);
      return false;
    }
    result = fillView_<ELEMENT>(bh);
    return true;
  }

  template <typename ELEMENT>
  bool Event::getByLabel(std::string const& moduleLabel, Handle<View<ELEMENT>>& result) const {
    return getByLabel(moduleLabel, emptyString_, result);
  }

  template <typename ELEMENT>
  bool Event::getByToken(EDGetToken token, Handle<View<ELEMENT>>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(ELEMENT)), ELEMENT_TYPE, token, moduleCallingContext_);
    if UNLIKELY (bh.failedToGet()) {
      Handle<View<ELEMENT>> h(std::move(bh.whyFailedFactory()));
      h.swap(result);
      return false;
    }
    result = fillView_<ELEMENT>(bh);
    return true;
  }

  template <typename ELEMENT>
  bool Event::getByToken(EDGetTokenT<View<ELEMENT>> token, Handle<View<ELEMENT>>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(ELEMENT)), ELEMENT_TYPE, token, moduleCallingContext_);
    if UNLIKELY (bh.failedToGet()) {
      Handle<View<ELEMENT>> h(std::move(bh.whyFailedFactory()));
      h.swap(result);
      return false;
    }
    result = fillView_<ELEMENT>(bh);
    return true;
  }

  template <typename ELEMENT>
  Handle<View<ELEMENT>> Event::getHandle(EDGetTokenT<View<ELEMENT>> token) const {
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(ELEMENT)), ELEMENT_TYPE, token, moduleCallingContext_);
    if UNLIKELY (bh.failedToGet()) {
      return Handle<View<ELEMENT>>(std::move(bh.whyFailedFactory()));
    }
    return fillView_<ELEMENT>(bh);
  }

  template <typename ELEMENT>
  View<ELEMENT> const& Event::get(EDGetTokenT<View<ELEMENT>> token) const noexcept(false) {
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(ELEMENT)), ELEMENT_TYPE, token, moduleCallingContext_);
    if UNLIKELY (bh.failedToGet()) {
      bh.whyFailedFactory()->make()->raise();
    }
    return *fillView_<ELEMENT>(bh);
  }

  template <typename ELEMENT>
  Handle<View<ELEMENT>> Event::fillView_(BasicHandle& bh) const {
    std::vector<void const*> pointersToElements;
    FillViewHelperVector helpers;
    // the following must initialize the
    //  fill the helper vector
    bh.wrapper()->fillView(bh.id(), pointersToElements, helpers);

    auto newview = std::make_shared<View<ELEMENT>>(pointersToElements, helpers, &(productGetter()));

    addToGotBranchIDs(*bh.provenance());
    gotViews_.push_back(newview);
    return Handle<View<ELEMENT>>(newview.get(), bh.provenance());
  }

  // Free functions to retrieve a collection from the Event.
  // Will throw an exception if the collection is not available.

  template <typename T>
  T const& get(Event const& event, InputTag const& tag) noexcept(false) {
    Handle<T> handle;
    event.getByLabel(tag, handle);
    // throw if the handle is not valid
    return *handle.product();
  }

  template <typename T>
  T const& get(Event const& event, EDGetToken const& token) noexcept(false) {
    Handle<T> handle;
    event.getByToken(token, handle);
    // throw if the handle is not valid
    return *handle.product();
  }

  template <typename T>
  T const& get(Event const& event, EDGetTokenT<T> const& token) noexcept(false) {
    return event.get(token);
  }

}  // namespace edm

#endif  // FWCore_Framework_Event_h
