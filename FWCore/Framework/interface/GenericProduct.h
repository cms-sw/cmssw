#include <cstring>
#include <memory>
#include <typeinfo>
#include <utility>

#include "DataFormats/Common/interface/OrphanHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"

namespace edm {

  class GenericProduct {
  public:
    // TODO make private and add accessors and a constructor
    TypeWithDict wrappedType_;
    ObjectWithDict object_;
  };

  template <>
  class OrphanHandle<GenericProduct> : public OrphanHandleBase {
  public:
    OrphanHandle(ObjectWithDict prod, ProductID const& id) : OrphanHandleBase(prod.address(), id) {}
  };

  // specialise Event::putImpl for GenericProduct
  template <>
  inline OrphanHandle<GenericProduct> Event::putImpl(EDPutToken::value_type index,
                                                     std::unique_ptr<GenericProduct> product) {
    /* TODO implement for ObjectWithDict
    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    if constexpr (not std::derived_from<T, DoNotSortUponInsertion> and requires(T& p) { p.post_insert(); }) {
        iProduct.post_insert();
    }
     */

    assert(index < putProducts().size());

    // construct a wrapper for the product
    ObjectWithDict wrapper = product->wrappedType_.construct();
    ObjectWithDict base = wrapper.castObject(TypeWithDict(typeid(WrapperBase)));
    std::unique_ptr<WrapperBase> wp(reinterpret_cast<WrapperBase*>(base.address()));

    // move the product into the wrapper and mark it as present
    wp->moveFrom(product->object_.address(), product->object_.typeOf().typeInfo());

    // move the wrapped product into the event
    putProducts()[index] = std::move(wp);

    // construct and return a handle to the product
    ObjectWithDict prod(product->object_.typeOf(), wrapper.get("obj").address());
    auto const& prodID = provRecorder_.getProductID(index);
    return OrphanHandle<GenericProduct>(prod, prodID);
  }

  // specialise Event::put for GenericProduct
  template <>
  inline OrphanHandle<GenericProduct> Event::put(EDPutToken token, std::unique_ptr<GenericProduct> product) {
    if (UNLIKELY(product.get() == nullptr)) {  // null pointer is illegal
      TypeID typeID(typeid(GenericProduct));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Event", typeID, provRecorder_.productInstanceLabel(token));
    }
    std::type_info const& type = product->object_.typeOf().typeInfo();
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
