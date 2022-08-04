#ifndef FWCore_Framework_Run_h
#define FWCore_Framework_Run_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     Run
//
/**\class Run Run.h FWCore/Framework/interface/Run.h

Description: This is the primary interface for accessing per run EDProducts and inserting new derived products.

For its usage, see "FWCore/Framework/interface/PrincipalGetAdapter.h"

*/
/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Common/interface/RunBase.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/Likely.h"

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace edm {
  class ModuleCallingContext;
  class ProducerBase;
  class SharedResourcesAcquirer;

  namespace stream {
    template <typename T>
    class ProducingModuleAdaptorBase;
  }

  class Run : public RunBase {
  public:
    Run(RunTransitionInfo const&, ModuleDescription const&, ModuleCallingContext const*, bool isAtEnd);
    Run(RunPrincipal const&, ModuleDescription const&, ModuleCallingContext const*, bool isAtEnd);
    ~Run() override;

    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer) { provRecorder_.setConsumer(iConsumer); }

    void setSharedResourcesAcquirer(SharedResourcesAcquirer* iResourceAcquirer) {
      provRecorder_.setSharedResourcesAcquirer(iResourceAcquirer);
    }

    void setProducer(ProducerBase const* iProducer);

    typedef PrincipalGetAdapter Base;
    // AUX functions are defined in RunBase
    RunAuxiliary const& runAuxiliary() const final;
    // AUX functions.
    //     RunID const& id() const {return aux_.id();}
    //     RunNumber_t run() const {return aux_.run();}
    //     Timestamp const& beginTime() const {return aux_.beginTime();}
    //     Timestamp const& endTime() const {return aux_.endTime();}

    /**\return Reusable index which can be used to separate data for different simultaneous Runs.
     */
    RunIndex index() const;

    /**If you are caching data from the Run, you should also keep
     this number.  If this number changes then you know that
     the data you have cached is invalid.
     The value of '0' will never be returned so you can use that to
     denote that you have not yet checked the value.
     */
    typedef unsigned long CacheIdentifier_t;
    CacheIdentifier_t cacheIdentifier() const;

    template <typename PROD>
    bool getByLabel(std::string const& label, Handle<PROD>& result) const;

    template <typename PROD>
    bool getByLabel(std::string const& label, std::string const& productInstanceName, Handle<PROD>& result) const;

    /// same as above, but using the InputTag class
    template <typename PROD>
    bool getByLabel(InputTag const& tag, Handle<PROD>& result) const;

    template <typename PROD>
    bool getByToken(EDGetToken token, Handle<PROD>& result) const;

    template <typename PROD>
    bool getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const;

    template <typename PROD>
    Handle<PROD> getHandle(EDGetTokenT<PROD> token) const;

    template <typename PROD>
    PROD const& get(EDGetTokenT<PROD> token) const noexcept(false);

    template <typename PROD>
    void getManyByType(std::vector<Handle<PROD>>& results) const;

    ///Put a new product.
    template <typename PROD>
    void put(std::unique_ptr<PROD> product) {
      put<PROD>(std::move(product), std::string());
    }

    ///Put a new product with a 'product instance name'
    template <typename PROD>
    void put(std::unique_ptr<PROD> product, std::string const& productInstanceName);

    template <typename PROD>
    void put(EDPutToken token, std::unique_ptr<PROD> product);

    template <typename PROD>
    void put(EDPutTokenT<PROD> token, std::unique_ptr<PROD> product);

    ///puts a new product
    template <typename PROD, typename... Args>
    void emplace(EDPutTokenT<PROD> token, Args&&... args);

    template <typename PROD, typename... Args>
    void emplace(EDPutToken token, Args&&... args);

    Provenance const& getProvenance(BranchID const& theID) const;

    StableProvenance const& getStableProvenance(BranchID const& theID) const;

    void getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const;

    // Return true if this Run has been subjected to a process with
    // the given processName, and false otherwise.
    // If true is returned, then ps is filled with the ParameterSets
    // (possibly more than one) used to configure the identified
    // process(es). Equivalent ParameterSets are compressed out of the
    // result.
    // This function is not yet implemented in full.
    //bool
    //getProcessParameterSet(std::string const& processName, std::vector<ParameterSet>& ps) const;

    ProcessHistoryID const& processHistoryID() const;

    ProcessHistory const& processHistory() const;

    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }

    void labelsForToken(EDGetToken const& iToken, ProductLabels& oLabels) const {
      provRecorder_.labelsForToken(iToken, oLabels);
    }

  private:
    RunPrincipal const& runPrincipal() const;

    // Override version from RunBase class
    BasicHandle getByLabelImpl(std::type_info const& iWrapperType,
                               std::type_info const& iProductType,
                               InputTag const& iTag) const final;

    template <typename PROD>
    void putImpl(EDPutToken::value_type token, std::unique_ptr<PROD> product);

    template <typename PROD, typename... Args>
    void emplaceImpl(EDPutToken::value_type token, Args&&... args);

    typedef std::vector<edm::propagate_const<std::unique_ptr<WrapperBase>>> ProductPtrVec;
    ProductPtrVec& putProducts() { return putProducts_; }
    ProductPtrVec const& putProducts() const { return putProducts_; }

    // commit_() is called to complete the transaction represented by
    // this PrincipalGetAdapter. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class RawInputSource;
    friend class ProducerBase;
    template <typename T>
    friend class stream::ProducingModuleAdaptorBase;

    void commit_(std::vector<edm::ProductResolverIndex> const& iShouldPut);

    PrincipalGetAdapter provRecorder_;
    ProductPtrVec putProducts_;
    RunAuxiliary const& aux_;
    ModuleCallingContext const* moduleCallingContext_;
    SharedResourcesAcquirer*
        sharedResourcesAcquirer_;  // We do not use propagate_const because the acquirer is itself mutable.

    static const std::string emptyString_;
  };

  template <typename PROD>
  void Run::putImpl(EDPutToken::value_type index, std::unique_ptr<PROD> product) {
    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    std::conditional_t<detail::has_postinsert<PROD>::value, DoPostInsert<PROD>, DoNotPostInsert<PROD>> maybe_inserter;
    maybe_inserter(product.get());

    assert(index < putProducts().size());

    std::unique_ptr<Wrapper<PROD>> wp(new Wrapper<PROD>(std::move(product)));
    putProducts()[index] = std::move(wp);
  }

  template <typename PROD>
  void Run::put(std::unique_ptr<PROD> product, std::string const& productInstanceName) {
    if (UNLIKELY(product.get() == nullptr)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Run", typeID, productInstanceName);
    }
    auto index = provRecorder_.getPutTokenIndex(TypeID(*product), productInstanceName);
    putImpl(index, std::move(product));
  }

  template <typename PROD>
  void Run::put(EDPutTokenT<PROD> token, std::unique_ptr<PROD> product) {
    if (UNLIKELY(product.get() == nullptr)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Run", typeID, provRecorder_.productInstanceLabel(token));
    }
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Run", typeid(PROD));
    }
    putImpl(token.index(), std::move(product));
  }

  template <typename PROD>
  void Run::put(EDPutToken token, std::unique_ptr<PROD> product) {
    if (UNLIKELY(product.get() == 0)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Run", typeID, provRecorder_.productInstanceLabel(token));
    }
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Run", typeid(PROD));
    }
    if (UNLIKELY(provRecorder_.getTypeIDForPutTokenIndex(token.index()) != TypeID{typeid(PROD)})) {
      principal_get_adapter_detail::throwOnPutOfWrongType(typeid(PROD),
                                                          provRecorder_.getTypeIDForPutTokenIndex(token.index()));
    }

    putImpl(token.index(), std::move(product));
  }

  template <typename PROD, typename... Args>
  void Run::emplace(EDPutTokenT<PROD> token, Args&&... args) {
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Run", typeid(PROD));
    }
    emplaceImpl<PROD>(token.index(), std::forward<Args>(args)...);
  }

  template <typename PROD, typename... Args>
  void Run::emplace(EDPutToken token, Args&&... args) {
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Run", typeid(PROD));
    }
    if (UNLIKELY(provRecorder_.getTypeIDForPutTokenIndex(token.index()) != TypeID{typeid(PROD)})) {
      principal_get_adapter_detail::throwOnPutOfWrongType(typeid(PROD),
                                                          provRecorder_.getTypeIDForPutTokenIndex(token.index()));
    }

    emplaceImpl(token.index(), std::forward<Args>(args)...);
  }

  template <typename PROD, typename... Args>
  void Run::emplaceImpl(EDPutToken::value_type index, Args&&... args) {
    assert(index < putProducts().size());

    std::unique_ptr<Wrapper<PROD>> wp(new Wrapper<PROD>(WrapperBase::Emplace{}, std::forward<Args>(args)...));

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    std::conditional_t<detail::has_postinsert<PROD>::value, DoPostInsert<PROD>, DoNotPostInsert<PROD>> maybe_inserter;
    maybe_inserter(&(wp->bareProduct()));

    putProducts()[index] = std::move(wp);
  }

  template <typename PROD>
  bool Run::getByLabel(std::string const& label, Handle<PROD>& result) const {
    return getByLabel(label, emptyString_, result);
  }

  template <typename PROD>
  bool Run::getByLabel(std::string const& label, std::string const& productInstanceName, Handle<PROD>& result) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), label, productInstanceName);
    }
    result.clear();
    BasicHandle bh = provRecorder_.getByLabel_(
        TypeID(typeid(PROD)), label, productInstanceName, emptyString_, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  /// same as above, but using the InputTag class
  template <typename PROD>
  bool Run::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), tag.label(), tag.instance());
    }
    result.clear();
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(typeid(PROD)), tag, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  bool Run::getByToken(EDGetToken token, Handle<PROD>& result) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), token);
    }
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  bool Run::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), token);
    }
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  Handle<PROD> Run::getHandle(EDGetTokenT<PROD> token) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), token);
    }
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    return convert_handle<PROD>(std::move(bh));
  }

  template <typename PROD>
  PROD const& Run::get(EDGetTokenT<PROD> token) const noexcept(false) {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), token);
    }
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    return *convert_handle<PROD>(std::move(bh));
  }

  template <typename PROD>
  void Run::getManyByType(std::vector<Handle<PROD>>& results) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)));
    }
    return provRecorder_.getManyByType(results, moduleCallingContext_);
  }

  // Free functions to retrieve a collection from the Run.
  // Will throw an exception if the collection is not available.

  template <typename T>
  T const& get(Run const& event, InputTag const& tag) {
    Handle<T> handle;
    event.getByLabel(tag, handle);
    // throw if the handle is not valid
    return *handle.product();
  }

  template <typename T>
  T const& get(Run const& event, EDGetToken const& token) {
    Handle<T> handle;
    event.getByToken(token, handle);
    // throw if the handle is not valid
    return *handle.product();
  }

  template <typename T>
  T const& get(Run const& event, EDGetTokenT<T> const& token) {
    Handle<T> handle;
    event.getByToken(token, handle);
    // throw if the handle is not valid
    return *handle.product();
  }

}  // namespace edm

#endif  // FWCore_Framework_Run_h
