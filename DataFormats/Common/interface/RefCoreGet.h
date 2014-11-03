#ifndef DataFormats_Common_RefCoreGet_h
#define DataFormats_Common_RefCoreGet_h

/*----------------------------------------------------------------------

RefCoreGet: Free function to get the pointer to a referenced product.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace edm {

  namespace refcore {
    template <typename T>
    inline
    T const*
    getProductPtr_(RefCore const& ref) {
      //if (isNull()) throwInvalidReference();
      assert (!ref.isTransient());
      WrapperBase const* product = ref.getProductPtr(typeid(T));
      Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(product);
      if (wrapper == nullptr) { 	 
        ref.wrongTypeException(typeid(T), typeid(*product)); 	 
      }
      ref.setProductPtr(wrapper->product());
      return wrapper->product();
    }
  }

  template <typename T>
  inline
  T const*
  getProduct(RefCore const& ref) {
    T const* p = static_cast<T const*>(ref.productPtr());
    if (p != 0) return p;
    if (ref.isTransient()) {
      ref.nullPointerForTransientException(typeid(T));
    }
    return refcore::getProductPtr_<T>(ref);
  }

  namespace refcore {
    template <typename T>
    inline
    T const*
    tryToGetProductPtr_(RefCore const& ref) {
      //if (isNull()) throwInvalidReference();
      assert (!ref.isTransient());
      WrapperBase const* product = ref.tryToGetProductPtr(typeid(T));
      if(product == nullptr) {
        return nullptr;
      }
      Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(product);
      ref.setProductPtr(wrapper->product());
      return wrapper->product();
    }
  }

  template <typename T>
  inline
  T const*
  tryToGetProduct(RefCore const& ref) {
    T const* p = static_cast<T const*>(ref.productPtr());
    if (p != 0) return p;
    if (ref.isTransient()) {
      ref.nullPointerForTransientException(typeid(T));
    }
    return refcore::tryToGetProductPtr_<T>(ref);
  }

  namespace refcore {
    template <typename T>
    inline
    T const*
    getThinnedProductPtr_(RefCore const& ref, unsigned int& thinnedKey) {
      assert (!ref.isTransient());
      WrapperBase const* product = ref.getThinnedProductPtr(typeid(T), thinnedKey);
      Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(product);
      // Do not cache the pointer to a thinned collection
      // ref.setProductPtr(wrapper->product());
      return wrapper->product();
    }
  }

  template <typename T>
  inline
  T const*
  getThinnedProduct(RefCore const& ref, unsigned int& thinnedKey) {
    // The pointer to a thinned collection will never be cached
    // T const* p = static_cast<T const*>(ref.productPtr());
    // if (p != 0) return p;

    if (ref.isTransient()) {
      ref.nullPointerForTransientException(typeid(T));
    }
    return refcore::getThinnedProductPtr_<T>(ref, thinnedKey);
  }
}
#endif
