#ifndef DataFormats_Common_Wrapper_h
#define DataFormats_Common_Wrapper_h

/*----------------------------------------------------------------------

Wrapper: A template wrapper around EDProducts to hold the product ID.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/WrapperDetail.h"
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include "FWCore/Utilities/interface/Visibility.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <typeinfo>

namespace edm {
  template<typename T>
  class Wrapper : public WrapperBase {
  public:
    typedef T value_type;
    typedef T wrapped_type; // used with the dictionary to identify Wrappers
    Wrapper() : WrapperBase(), present(false), obj() {}
#ifndef __GCCXML__
    explicit Wrapper(std::unique_ptr<T> ptr);
#endif
    virtual ~Wrapper() {}
    T const* product() const {return (present ? &obj : 0);}
    T const* operator->() const {return product();}

    //these are used by FWLite
    static std::type_info const& productTypeInfo() {return typeid(T);}
    static std::type_info const& typeInfo() {return typeid(Wrapper<T>);}

    // the constructor takes ownership of T*
    Wrapper(T*);

    //Used by ROOT storage
    CMS_CLASS_VERSION(3)

private:
    virtual bool isPresent_() const GCC11_OVERRIDE {return present;}
    virtual std::type_info const& dynamicTypeInfo_() const GCC11_OVERRIDE {return typeid(T);}
    virtual std::type_info const& wrappedTypeInfo_() const GCC11_OVERRIDE {return typeid(Wrapper<T>);}

    virtual std::type_info const& valueTypeInfo_() const GCC11_OVERRIDE;
    virtual std::type_info const& memberTypeInfo_() const GCC11_OVERRIDE;
#ifndef __GCCXML__
    virtual bool isMergeable_() const GCC11_OVERRIDE;
    virtual bool mergeProduct_(WrapperBase const* newProduct) GCC11_OVERRIDE;
    virtual bool hasIsProductEqual_() const GCC11_OVERRIDE;
    virtual bool isProductEqual_(WrapperBase const* newProduct) const GCC11_OVERRIDE;
#endif

    virtual void do_fillView(ProductID const& id,
                             std::vector<void const*>& pointers,
                             FillViewHelperVector& helpers) const GCC11_OVERRIDE;
    virtual void do_setPtr(std::type_info const& iToType,
                           unsigned long iIndex,
                           void const*& oPtr) const GCC11_OVERRIDE;
    virtual void do_fillPtrVector(std::type_info const& iToType,
                                  std::vector<unsigned long> const& iIndices,
                                  std::vector<void const*>& oPtr) const GCC11_OVERRIDE;

  private:
    // We wish to disallow copy construction and assignment.
    // We make the copy constructor and assignment operator private.
    Wrapper(Wrapper<T> const& rh); // disallow copy construction
    Wrapper<T>& operator=(Wrapper<T> const&); // disallow assignment

    bool present;
    T obj;
  };

#ifndef __GCCXML__
  template<typename T>
  inline
  void swap_or_assign(T& a, T& b) {
    detail::doSwapOrAssign<T>()(a, b);
  } 

  template<typename T>
  Wrapper<T>::Wrapper(std::unique_ptr<T> ptr) :
    WrapperBase(),
    present(ptr.get() != 0),
    obj() {
    if (present) {
      // The following will call swap if T has such a function,
      // and use assignment if T has no such function.
      swap_or_assign(obj, *ptr);
    }
  }

  template<typename T>
  Wrapper<T>::Wrapper(T* ptr) :
  WrapperBase(),
  present(ptr != 0),
  obj() {
     std::unique_ptr<T> temp(ptr);
     if (present) {
        // The following will call swap if T has such a function,
        // and use assignment if T has no such function.
        swap_or_assign(obj, *ptr);
     }
  }

  template<typename T>
  inline
  std::type_info const& Wrapper<T>::valueTypeInfo_() const {
    return detail::getValueType<T>()();
  }

  template<typename T>
  inline
  std::type_info const& Wrapper<T>::memberTypeInfo_() const {
    return detail::getMemberType<T>()();
  }

  template<typename T>
  inline 
  bool Wrapper<T>::isMergeable_() const {
    return detail::getHasMergeFunction<T>()();
  }

  template<typename T>
  inline
  bool Wrapper<T>::mergeProduct_(WrapperBase const* newProduct) {
    Wrapper<T> const* wrappedNewProduct = dynamic_cast<Wrapper<T> const*>(newProduct);
    assert(wrappedNewProduct != nullptr);
    return detail::doMergeProduct<T>()(obj, wrappedNewProduct->obj);
  }

  template<typename T>
  inline
  bool Wrapper<T>::hasIsProductEqual_() const {
    return detail::getHasIsProductEqual<T>()();
  }

  template<typename T>
  inline
  bool Wrapper<T>::isProductEqual_(WrapperBase const* newProduct) const {
    Wrapper<T> const* wrappedNewProduct = dynamic_cast<Wrapper<T> const*>(newProduct);
    assert(wrappedNewProduct != nullptr);
    return detail::doIsProductEqual<T>()(obj, wrappedNewProduct->obj);
  }
#endif
}

#include "DataFormats/Common/interface/WrapperView.icc"

#endif
