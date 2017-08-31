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
    explicit Wrapper(std::unique_ptr<T> ptr);
    ~Wrapper() override {}
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
    bool isPresent_() const override {return present;}
    std::type_info const& dynamicTypeInfo_() const override {return typeid(T);}
    std::type_info const& wrappedTypeInfo_() const override {return typeid(Wrapper<T>);}

    std::type_info const& valueTypeInfo_() const override;
    std::type_info const& memberTypeInfo_() const override;
    bool isMergeable_() const override;
    bool mergeProduct_(WrapperBase const* newProduct) override;
    bool hasIsProductEqual_() const override;
    bool isProductEqual_(WrapperBase const* newProduct) const override;

    void do_fillView(ProductID const& id,
                             std::vector<void const*>& pointers,
                             FillViewHelperVector& helpers) const override;
    void do_setPtr(std::type_info const& iToType,
                           unsigned long iIndex,
                           void const*& oPtr) const override;
    void do_fillPtrVector(std::type_info const& iToType,
                                  std::vector<unsigned long> const& iIndices,
                                  std::vector<void const*>& oPtr) const override;

  private:
    // We wish to disallow copy construction and assignment.
    // We make the copy constructor and assignment operator private.
    Wrapper(Wrapper<T> const& rh) = delete; // disallow copy construction
    Wrapper<T>& operator=(Wrapper<T> const&) = delete; // disallow assignment

    bool present;
    T obj;
  };

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
}

#include "DataFormats/Common/interface/WrapperView.icc"

#endif
