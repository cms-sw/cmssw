#ifndef FWCore_Framework_LuminosityBlock_h
#define FWCore_Framework_LuminosityBlock_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     LuminosityBlock
//
/**\class LuminosityBlock LuminosityBlock.h FWCore/Framework/interface/LuminosityBlock.h

Description: This is the primary interface for accessing per luminosity block EDProducts
and inserting new derived per luminosity block EDProducts.

For its usage, see "FWCore/Framework/interface/PrincipalGetAdapter.h"

*/
/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Common/interface/LuminosityBlockBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>
#include <optional>

namespace edm {
  class ModuleCallingContext;
  class ProducerBase;
  class SharedResourcesAcquirer;

  namespace stream {
    template <typename T>
    class ProducingModuleAdaptorBase;
  }

  class LuminosityBlock : public LuminosityBlockBase {
  public:
    LuminosityBlock(LumiTransitionInfo const&, ModuleDescription const&, ModuleCallingContext const*, bool isAtEnd);
    LuminosityBlock(LuminosityBlockPrincipal const&,
                    ModuleDescription const&,
                    ModuleCallingContext const*,
                    bool isAtEnd);
    ~LuminosityBlock() override;

    // AUX functions are defined in LuminosityBlockBase
    LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const final;

    /**\return Reusable index which can be used to separate data for different simultaneous LuminosityBlocks.
     */
    LuminosityBlockIndex index() const;

    /**If you are caching data from the LuminosityBlock, you should also keep
     this number.  If this number changes then you know that
     the data you have cached is invalid.
     The value of '0' will never be returned so you can use that to
     denote that you have not yet checked the value.
     */
    typedef unsigned long CacheIdentifier_t;
    CacheIdentifier_t cacheIdentifier() const;

    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer);

    void setSharedResourcesAcquirer(SharedResourcesAcquirer* iResourceAcquirer);

    void setProducer(ProducerBase const* iProducer);

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

    Run const& getRun() const {
      if (not run_) {
        fillRun();
      }
      return run_.value();
    }

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

    ProcessHistoryID const& processHistoryID() const;

    ProcessHistory const& processHistory() const;

    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }

    void labelsForToken(EDGetToken const& iToken, ProductLabels& oLabels) const {
      provRecorder_.labelsForToken(iToken, oLabels);
    }

  private:
    LuminosityBlockPrincipal const& luminosityBlockPrincipal() const;

    // Override version from LuminosityBlockBase class
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

    void fillRun() const;

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
    LuminosityBlockAuxiliary const& aux_;
    //This class is intended to be used by only one thread
    CMS_SA_ALLOW mutable std::optional<Run> run_;
    ModuleCallingContext const* moduleCallingContext_;

    static const std::string emptyString_;
  };

  template <typename PROD>
  void LuminosityBlock::putImpl(EDPutToken::value_type index, std::unique_ptr<PROD> product) {
    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    std::conditional_t<detail::has_postinsert<PROD>::value, DoPostInsert<PROD>, DoNotPostInsert<PROD>> maybe_inserter;
    maybe_inserter(product.get());

    assert(index < putProducts().size());

    std::unique_ptr<Wrapper<PROD>> wp(new Wrapper<PROD>(std::move(product)));
    putProducts()[index] = std::move(wp);
  }

  template <typename PROD>
  void LuminosityBlock::put(std::unique_ptr<PROD> product, std::string const& productInstanceName) {
    if (UNLIKELY(product.get() == nullptr)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("LuminosityBlock", typeID, productInstanceName);
    }
    auto index = provRecorder_.getPutTokenIndex(TypeID(*product), productInstanceName);
    putImpl(index, std::move(product));
  }

  template <typename PROD>
  void LuminosityBlock::put(EDPutTokenT<PROD> token, std::unique_ptr<PROD> product) {
    if (UNLIKELY(product.get() == nullptr)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct(
          "LuminosityBlock", typeID, provRecorder_.productInstanceLabel(token));
    }
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("LuminosityBlock", typeid(PROD));
    }
    putImpl(token.index(), std::move(product));
  }

  template <typename PROD>
  void LuminosityBlock::put(EDPutToken token, std::unique_ptr<PROD> product) {
    if (UNLIKELY(product.get() == nullptr)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct(
          "LuminosityBlock", typeID, provRecorder_.productInstanceLabel(token));
    }
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("LuminosityBlock", typeid(PROD));
    }
    if (UNLIKELY(provRecorder_.getTypeIDForPutTokenIndex(token.index()) != TypeID{typeid(PROD)})) {
      principal_get_adapter_detail::throwOnPutOfWrongType(typeid(PROD),
                                                          provRecorder_.getTypeIDForPutTokenIndex(token.index()));
    }

    putImpl(token.index(), std::move(product));
  }

  template <typename PROD, typename... Args>
  void LuminosityBlock::emplace(EDPutTokenT<PROD> token, Args&&... args) {
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("LuminosityBlock", typeid(PROD));
    }
    emplaceImpl<PROD>(token.index(), std::forward<Args>(args)...);
  }

  template <typename PROD, typename... Args>
  void LuminosityBlock::emplace(EDPutToken token, Args&&... args) {
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("LuminosityBlock", typeid(PROD));
    }
    if (UNLIKELY(provRecorder_.getTypeIDForPutTokenIndex(token.index()) != TypeID{typeid(PROD)})) {
      principal_get_adapter_detail::throwOnPutOfWrongType(typeid(PROD),
                                                          provRecorder_.getTypeIDForPutTokenIndex(token.index()));
    }

    emplaceImpl(token.index(), std::forward<Args>(args)...);
  }

  template <typename PROD, typename... Args>
  void LuminosityBlock::emplaceImpl(EDPutToken::value_type index, Args&&... args) {
    assert(index < putProducts().size());

    std::unique_ptr<Wrapper<PROD>> wp(new Wrapper<PROD>(WrapperBase::Emplace{}, std::forward<Args>(args)...));

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    std::conditional_t<detail::has_postinsert<PROD>::value, DoPostInsert<PROD>, DoNotPostInsert<PROD>> maybe_inserter;
    maybe_inserter(&(wp->bareProduct()));

    putProducts()[index] = std::move(wp);
  }

  template <typename PROD>
  bool LuminosityBlock::getByLabel(std::string const& label, Handle<PROD>& result) const {
    return getByLabel(label, emptyString_, result);
  }

  template <typename PROD>
  bool LuminosityBlock::getByLabel(std::string const& label,
                                   std::string const& productInstanceName,
                                   Handle<PROD>& result) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), label, productInstanceName);
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
  bool LuminosityBlock::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), tag.label(), tag.instance());
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
  bool LuminosityBlock::getByToken(EDGetToken token, Handle<PROD>& result) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), token);
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
  bool LuminosityBlock::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), token);
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
  Handle<PROD> LuminosityBlock::getHandle(EDGetTokenT<PROD> token) const {
    if UNLIKELY (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), token);
    }
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    return convert_handle<PROD>(std::move(bh));
  }

  template <typename PROD>
  PROD const& LuminosityBlock::get(EDGetTokenT<PROD> token) const noexcept(false) {
    if UNLIKELY (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), token);
    }
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    return *convert_handle<PROD>(std::move(bh));
  }

  template <typename PROD>
  void LuminosityBlock::getManyByType(std::vector<Handle<PROD>>& results) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)));
    }
    return provRecorder_.getManyByType(results, moduleCallingContext_);
  }

  // Free functions to retrieve a collection from the LuminosityBlock.
  // Will throw an exception if the collection is not available.

  template <typename T>
  T const& get(LuminosityBlock const& event, InputTag const& tag) {
    Handle<T> handle;
    event.getByLabel(tag, handle);
    // throw if the handle is not valid
    return *handle.product();
  }

  template <typename T>
  T const& get(LuminosityBlock const& event, EDGetToken const& token) {
    Handle<T> handle;
    event.getByToken(token, handle);
    // throw if the handle is not valid
    return *handle.product();
  }

  template <typename T>
  T const& get(LuminosityBlock const& event, EDGetTokenT<T> const& token) {
    Handle<T> handle;
    event.getByToken(token, handle);
    // throw if the handle is not valid
    return *handle.product();
  }

}  // namespace edm

#endif  // FWCore_Framework_LuminosityBlock_h
