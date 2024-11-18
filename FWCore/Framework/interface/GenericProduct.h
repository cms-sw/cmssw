#ifndef FWCore_Framework_interface_GenericProduct_h
#define FWCore_Framework_interface_GenericProduct_h

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
    GenericProduct(ObjectWithDict const& object, TypeWithDict const& wrapper)
        : object_{object}, wrapperType_{wrapper} {}

    // The GenericProduct *does not* take ownership of the object.
    // The caller must guarantee that the object remains valid as long as needed.
    template <typename T>
    GenericProduct(T& object) : object_{typeid(T), &object}, wrapperType_{typeid(Wrapper<T>)} {}

    ObjectWithDict const& object() const { return object_; }
    TypeWithDict const& wrapperType() const { return wrapperType_; }

  private:
    ObjectWithDict object_;
    TypeWithDict wrapperType_;
  };

  template <>
  class OrphanHandle<GenericProduct> : public OrphanHandleBase {
  public:
    OrphanHandle(ObjectWithDict prod, ProductID const& id) : OrphanHandleBase(prod.address(), id), prod_(prod) {}

    ObjectWithDict const* product() const { return &prod_; }

    ObjectWithDict const* operator->() const { return &prod_; }

    ObjectWithDict const& operator*() const { return prod_; }

  private:
    ObjectWithDict prod_;
  };

  // specialise Event::putImpl for GenericProduct
  template <>
  inline OrphanHandle<GenericProduct> Event::putImpl(EDPutToken::value_type index,
                                                     std::unique_ptr<GenericProduct> product) {
    static const TypeWithDict classDoNotSortUponInsertion(typeid(DoNotSortUponInsertion));
    static const TypeWithDict classWrapperBase(typeid(WrapperBase));

    assert(index < putProducts().size());

    // construct a wrapper for the product
    ObjectWithDict const& wrapper = product->wrapperType().construct();
    ObjectWithDict const& base = wrapper.castObject(classWrapperBase);
    std::unique_ptr<WrapperBase> wp(reinterpret_cast<WrapperBase*>(base.address()));

    // move the product into the wrapper and mark it as present
    // internally this calls post_insert, if the underlying type has such a function
    wp->moveFrom(product->object().address(), product->object().typeOf().typeInfo());

    // move the wrapped product into the event
    putProducts()[index] = std::move(wp);

    // construct and return a handle to the product
    ObjectWithDict prod(product->object().typeOf(), wrapper.get("obj").address());
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
    std::type_info const& type = product->object().typeOf().typeInfo();
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

#endif  // FWCore_Framework_interface_GenericProduct_h
