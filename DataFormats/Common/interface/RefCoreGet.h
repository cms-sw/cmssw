#ifndef DataFormats_Common_RefCoreGet_h
#define DataFormats_Common_RefCoreGet_h

/*----------------------------------------------------------------------

RefCoreGet: Free function to get the pointer to a referenced product.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <cassert>
#include <typeinfo>

namespace edm {

  namespace refcore {
    template <typename T>
    inline
    T const*
    getProduct_(RefCore const& ref, const EDProductGetter* prodGetter) {
      assert (!ref.isTransient());
      WrapperBase const* product = ref.getProductPtr(typeid(T),prodGetter);
      Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(product);
      if (wrapper == nullptr) { 	 
        ref.wrongTypeException(typeid(T), typeid(*product)); 	 
      }
      ref.setProductPtr(wrapper->product());
      return wrapper->product();
    }
  }

  // Get the product using a RefCore from a RefProd.
  // In this case the pointer cache in the RefCore
  // is designed to hold a pointer to the container product.
  template <typename T>
  inline
  T const*
  getProduct(RefCore const& ref) {
    T const* p = static_cast<T const*>(ref.productPtr());
    if (p != nullptr) return p;
    if (ref.isTransient()) {
      ref.nullPointerForTransientException(typeid(T));
    }
    auto productGetter = ref.productGetter();
    if(nullptr == productGetter) {
      p =static_cast<T const*>(ref.productPtr());
      if(p != nullptr) {
        //another thread updated the value since we checked
        return p;
      }
    }
    return refcore::getProduct_<T>(ref, productGetter);
  }

  namespace refcore {
    template <typename T>
    inline
    T const*
    getProductWithCoreFromRef_(RefCore const& ref, EDProductGetter const* prodGetter) {
      WrapperBase const* product = ref.getProductPtr(typeid(T), prodGetter);
      Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(product);
      if (wrapper == nullptr) {
        ref.wrongTypeException(typeid(T), typeid(*product));
      }
      return wrapper->product();
    }
  }

  // Get the product using a RefCore from a Ref.
  // In this case the pointer cache in the RefCore
  // is designed to hold a pointer to an element in the container product.
  template <typename T>
  inline
  T const*
  getProductWithCoreFromRef(RefCore const& ref, EDProductGetter const* prodGetter) {
    if (ref.isTransient()) {
      ref.nullPointerForTransientException(typeid(T));
    }
    return refcore::getProductWithCoreFromRef_<T>(ref, prodGetter);
  }

  namespace refcore {
    template <typename T>
    inline
    T const*
    tryToGetProductWithCoreFromRef_(RefCore const& ref, EDProductGetter const* prodGetter) {
      WrapperBase const* product = ref.tryToGetProductPtr(typeid(T),prodGetter);
      if(product == nullptr) {
        return nullptr;
      }
      Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(product);
      return wrapper->product();
    }
  }

  // get the product using a RefCore from a Ref
  // In this case the pointer cache in the RefCore
  // is for a pointer to an element in the container product.
  // In this case we try only, which means we do not throw
  // if we fail. This gives the calling function the
  // chance to look in thinned containers.
  template <typename T>
  inline
  T const*
  tryToGetProductWithCoreFromRef(RefCore const& ref, EDProductGetter const* prodGetter) {
    if (ref.isTransient()) {
      ref.nullPointerForTransientException(typeid(T));
    }
    return refcore::tryToGetProductWithCoreFromRef_<T>(ref, prodGetter);
  }

  namespace refcore {
    template <typename T>
    inline
    T const*
    getThinnedProduct_(RefCore const& ref, unsigned int& thinnedKey, EDProductGetter const* prodGetter) {
      WrapperBase const* product = ref.getThinnedProductPtr(typeid(T), thinnedKey, prodGetter);
      Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(product);
      return wrapper->product();
    }
  }

  template <typename T>
  inline
  T const*
  getThinnedProduct(RefCore const& ref, unsigned int& thinnedKey, EDProductGetter const* prodGetter) {
    // The pointer to a thinned collection will never be cached
    // T const* p = static_cast<T const*>(ref.productPtr());
    // if (p != 0) return p;

    if (ref.isTransient()) {
      ref.nullPointerForTransientException(typeid(T));
    }
    return refcore::getThinnedProduct_<T>(ref, thinnedKey, prodGetter);
  }
}
#endif
