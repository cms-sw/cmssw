#ifndef FWCore_Framework_ProcessBlock_h
#define FWCore_Framework_ProcessBlock_h

/** \class edm::ProcessBlock

\author W. David Dagenhart, created 19 March, 2020

*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace edm {

  class ModuleCallingContext;
  class ModuleDescription;
  class ProcessBlockPrincipal;
  class ProducerBase;

  namespace stream {
    template <typename T>
    class ProducingModuleAdaptorBase;
  }

  class ProcessBlock {
  public:
    ProcessBlock(ProcessBlockPrincipal const&, ModuleDescription const&, ModuleCallingContext const*, bool isAtEnd);

    template <typename PROD>
    bool getByToken(EDGetToken token, Handle<PROD>& result) const;

    template <typename PROD>
    bool getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const;

    template <typename PROD>
    Handle<PROD> getHandle(EDGetTokenT<PROD> token) const;

    template <typename PROD>
    PROD const& get(EDGetTokenT<PROD> token) const noexcept(false);

    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer) { provRecorder_.setConsumer(iConsumer); }

    void setProducer(ProducerBase const* iProducer);

    /**If you are caching data from the ProcessBlock, you should also keep
     this number.  If this number changes then you know that
     the data you have cached is invalid.
     The value of '0' will never be returned so you can use that to
     denote that you have not yet checked the value.
     */
    using CacheIdentifier_t = unsigned long;
    CacheIdentifier_t cacheIdentifier() const;

    template <typename PROD>
    void put(EDPutTokenT<PROD> token, std::unique_ptr<PROD> product);

    template <typename PROD>
    void put(EDPutToken token, std::unique_ptr<PROD> product);

    template <typename PROD, typename... Args>
    void emplace(EDPutTokenT<PROD> token, Args&&... args);

    template <typename PROD, typename... Args>
    void emplace(EDPutToken token, Args&&... args);

    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }

  private:
    ProcessBlockPrincipal const& processBlockPrincipal() const;

    template <typename PROD>
    void putImpl(EDPutToken::value_type token, std::unique_ptr<PROD> product);

    template <typename PROD, typename... Args>
    void emplaceImpl(EDPutToken::value_type token, Args&&... args);

    friend class ProducerBase;
    template <typename T>
    friend class stream::ProducingModuleAdaptorBase;

    void commit_(std::vector<edm::ProductResolverIndex> const& iShouldPut);

    using ProductPtrVec = std::vector<edm::propagate_const<std::unique_ptr<WrapperBase>>>;
    ProductPtrVec& putProducts() { return putProducts_; }
    ProductPtrVec const& putProducts() const { return putProducts_; }

    PrincipalGetAdapter provRecorder_;
    ProductPtrVec putProducts_;
    ModuleCallingContext const* moduleCallingContext_;
  };

  template <typename PROD>
  bool ProcessBlock::getByToken(EDGetToken token, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  bool ProcessBlock::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  Handle<PROD> ProcessBlock::getHandle(EDGetTokenT<PROD> token) const {
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    return convert_handle<PROD>(std::move(bh));
  }

  template <typename PROD>
  PROD const& ProcessBlock::get(EDGetTokenT<PROD> token) const noexcept(false) {
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    return *convert_handle<PROD>(std::move(bh));
  }

  template <typename PROD>
  void ProcessBlock::put(EDPutTokenT<PROD> token, std::unique_ptr<PROD> product) {
    if (UNLIKELY(product.get() == 0)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct(
          "ProcessBlock", typeID, provRecorder_.productInstanceLabel(token));
    }
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("ProcessBlock", typeid(PROD));
    }
    putImpl(token.index(), std::move(product));
  }

  template <typename PROD>
  void ProcessBlock::put(EDPutToken token, std::unique_ptr<PROD> product) {
    if (UNLIKELY(product.get() == 0)) {  // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct(
          "ProcessBlock", typeID, provRecorder_.productInstanceLabel(token));
    }
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("ProcessBlock", typeid(PROD));
    }
    if (UNLIKELY(provRecorder_.getTypeIDForPutTokenIndex(token.index()) != TypeID{typeid(PROD)})) {
      principal_get_adapter_detail::throwOnPutOfWrongType(typeid(PROD),
                                                          provRecorder_.getTypeIDForPutTokenIndex(token.index()));
    }
    putImpl(token.index(), std::move(product));
  }

  template <typename PROD, typename... Args>
  void ProcessBlock::emplace(EDPutTokenT<PROD> token, Args&&... args) {
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("ProcessBlock", typeid(PROD));
    }
    emplaceImpl<PROD>(token.index(), std::forward<Args>(args)...);
  }

  template <typename PROD, typename... Args>
  void ProcessBlock::emplace(EDPutToken token, Args&&... args) {
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("ProcessBlock", typeid(PROD));
    }
    if (UNLIKELY(provRecorder_.getTypeIDForPutTokenIndex(token.index()) != TypeID{typeid(PROD)})) {
      principal_get_adapter_detail::throwOnPutOfWrongType(typeid(PROD),
                                                          provRecorder_.getTypeIDForPutTokenIndex(token.index()));
    }
    emplaceImpl<PROD>(token.index(), std::forward<Args>(args)...);
  }

  template <typename PROD>
  void ProcessBlock::putImpl(EDPutToken::value_type index, std::unique_ptr<PROD> product) {
    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    std::conditional_t<detail::has_postinsert<PROD>::value, DoPostInsert<PROD>, DoNotPostInsert<PROD>> maybe_inserter;
    maybe_inserter(product.get());

    assert(index < putProducts().size());

    std::unique_ptr<Wrapper<PROD>> wp(new Wrapper<PROD>(std::move(product)));
    putProducts()[index] = std::move(wp);
  }

  template <typename PROD, typename... Args>
  void ProcessBlock::emplaceImpl(EDPutToken::value_type index, Args&&... args) {
    assert(index < putProducts().size());

    std::unique_ptr<Wrapper<PROD>> wp(new Wrapper<PROD>(WrapperBase::Emplace{}, std::forward<Args>(args)...));

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    std::conditional_t<detail::has_postinsert<PROD>::value, DoPostInsert<PROD>, DoNotPostInsert<PROD>> maybe_inserter;
    maybe_inserter(&(wp->bareProduct()));

    putProducts()[index] = std::move(wp);
  }

  template <typename T>
  T const& get(ProcessBlock const& processBlock, EDGetToken const& token) {
    Handle<T> handle;
    processBlock.getByToken(token, handle);
    // throw if the handle is not valid
    return *handle.product();
  }

  template <typename T>
  T const& get(ProcessBlock const& processBlock, EDGetTokenT<T> const& token) {
    Handle<T> handle;
    processBlock.getByToken(token, handle);
    // throw if the handle is not valid
    return *handle.product();
  }

}  // namespace edm
#endif  // FWCore_Framework_ProcessBlock_h
