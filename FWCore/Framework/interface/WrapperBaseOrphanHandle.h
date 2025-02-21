#ifndef FWCore_Framework_interface_WrapperBaseOrphanHandle_h
#define FWCore_Framework_interface_WrapperBaseOrphanHandle_h

// c++ include files
#include <memory>
#include <typeinfo>

// CMSSW include files
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/TypeID.h"

// forward declarations
namespace edm {

  template <>
  class OrphanHandle<WrapperBase> {
  public:
    OrphanHandle() : product_(nullptr), id_() {}

    OrphanHandle(WrapperBase const* prod, ProductID const& id) : product_(prod), id_(id) { assert(product_); }

    // Reimplement the interface of OrphanHandleBase

    void clear() { product_ = nullptr; }

    bool isValid() const { return nullptr != product_; }

    ProductID id() const { return id_; }

    // Reimplement the interface of OrphanHandle

    WrapperBase const* product() const { return product_; }

    WrapperBase const* operator->() const { return this->product(); }
    WrapperBase const& operator*() const { return *(this->product()); }

  private:
    WrapperBase const* product_;
    ProductID id_;
  };

  // specialise Event::putImpl for WrapperBase
  template <>
  inline OrphanHandle<WrapperBase> Event::putImpl(EDPutToken::value_type index, std::unique_ptr<WrapperBase> product) {
    assert(index < putProducts().size());

    // move the wrapped product into the event
    putProducts()[index] = std::move(product);

    // construct and return a handle to the product
    WrapperBase const* prod = putProducts()[index].get();
    ProductID const& prodID = provRecorder_.getProductID(index);
    return OrphanHandle<WrapperBase>(prod, prodID);
  }

  // specialise Event::put for WrapperBase
  template <>
  inline OrphanHandle<WrapperBase> Event::put(EDPutToken token, std::unique_ptr<WrapperBase> product) {
    if (UNLIKELY(product.get() == nullptr)) {  // null pointer is illegal
      TypeID typeID(typeid(WrapperBase));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Event", typeID, provRecorder_.productInstanceLabel(token));
    }
    std::type_info const& type = product->dynamicTypeInfo();
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Event", type);
    }
    TypeID const& expected = provRecorder_.getTypeIDForPutTokenIndex(token.index());
    if (UNLIKELY(expected != TypeID{type})) {
      principal_get_adapter_detail::throwOnPutOfWrongType(type, expected);
    }

    return putImpl(token.index(), std::move(product));
  }

}  // namespace edm

#endif  // FWCore_Framework_interface_WrapperBaseOrphanHandle_h
