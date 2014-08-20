#ifndef DataFormats_Common_Wrapper_h
#define DataFormats_Common_Wrapper_h

/*----------------------------------------------------------------------

Wrapper: A template wrapper around EDProducts to hold the product ID.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/fwd_fillPtrVector.h"
#include "DataFormats/Common/interface/fwd_setPtr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include "FWCore/Utilities/interface/Visibility.h"

#include "boost/mpl/if.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <typeinfo>

namespace edm {
  template <typename T>
  class Wrapper : public EDProduct {
  public:
    typedef T value_type;
    typedef T wrapped_type;  // used with the dictionary to identify Wrappers
    Wrapper() : EDProduct(), present(false), obj() {}
    explicit Wrapper(std::auto_ptr<T> ptr);
    virtual ~Wrapper() {}
    T const* product() const {return (present ? &obj : 0);}
    T const* operator->() const {return product();}

    //these are used by FWLite
    static std::type_info const& productTypeInfo() {return typeid(T);}
    static std::type_info const& typeInfo() {return typeid(Wrapper<T>);}

    //  the constructor takes ownership of T*
    Wrapper(T*);

    //Used by ROOT storage
    CMS_CLASS_VERSION(3)

private:
    virtual bool isPresent_() const GCC11_OVERRIDE {return present;}
    virtual std::type_info const& dynamicTypeInfo_() const GCC11_OVERRIDE {return typeid(T);}
    virtual std::type_info const& wrappedTypeInfo_() const GCC11_OVERRIDE {return typeid(Wrapper<T>);}

#ifndef __GCCXML__
    virtual bool isMergeable_() const GCC11_OVERRIDE;
    virtual bool mergeProduct_(EDProduct const* newProduct) GCC11_OVERRIDE;
    virtual bool hasIsProductEqual_() const GCC11_OVERRIDE;
    virtual bool isProductEqual_(EDProduct const* newProduct) const GCC11_OVERRIDE;
#endif

    virtual void do_fillView(ProductID const& id,
                             std::vector<void const*>& pointers,
                             helper_vector_ptr& helpers) const GCC11_OVERRIDE;
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
    //   T const obj;
    T obj;
  };

} //namespace edm

#include "DataFormats/Common/interface/Ref.h"

namespace edm {

  template <typename T>
  struct DoFillView {
    void operator()(T const& obj,
                    ProductID const& id,
                    std::vector<void const*>& pointers,
                    helper_vector_ptr & helpers) const;
  };

  template <typename T>
  struct DoNotFillView {
    void operator()(T const&,
                    ProductID const&,
                    std::vector<void const*>&,
                    helper_vector_ptr&) const {
      Exception::throwThis(errors::ProductDoesNotSupportViews,
        "The product type ",
        typeid(T).name(),
        "\ndoes not support Views\n");
    }
  };

  template <typename T>
  inline
  void Wrapper<T>::do_fillView(ProductID const& id,
                               std::vector<void const*>& pointers,
                               helper_vector_ptr& helpers) const {
    typename boost::mpl::if_c<has_fillView<T>::value,
    DoFillView<T>,
    DoNotFillView<T> >::type maybe_filler;
    maybe_filler(obj, id, pointers, helpers);
  }


  template <typename T>
  struct DoSetPtr {
    void operator()(T const& obj,
                    std::type_info const& iToType,
                    unsigned long iIndex,
                    void const*& oPtr) const;
    void operator()(T const& obj,
                    std::type_info const& iToType,
                    std::vector<unsigned long> const& iIndex,
                    std::vector<void const*>& oPtr) const;
  };

  template <typename T>
  struct DoNotSetPtr {
    void operator()(T const& /*obj*/,
                    std::type_info const& /*iToType*/,
                    unsigned long /*iIndex*/,
                    void const*& /*oPtr*/) const {
      Exception::throwThis(errors::ProductDoesNotSupportPtr,
        "The product type ",
        typeid(T).name(),
        "\ndoes not support edm::Ptr\n");
    }
    void operator()(T const& /*obj*/,
                    std::type_info const& /*iToType*/,
                    std::vector<unsigned long> const& /*iIndexes*/,
                    std::vector<void const*>& /*oPtrs*/) const {
      Exception::throwThis(errors::ProductDoesNotSupportPtr,
        "The product type ",
        typeid(T).name(),
        "\ndoes not support edm::PtrVector\n");
    }
  };

  template <typename T>
  inline
  void Wrapper<T>::do_setPtr(std::type_info const& iToType,
                             unsigned long iIndex,
                             void const*& oPtr) const {
    typename boost::mpl::if_c<has_setPtr<T>::value,
    DoSetPtr<T>,
    DoNotSetPtr<T> >::type maybe_filler;
    maybe_filler(this->obj, iToType, iIndex, oPtr);
  }

  template <typename T>
  void Wrapper<T>::do_fillPtrVector(std::type_info const& iToType,
                                       std::vector<unsigned long> const& iIndices,
                                       std::vector<void const*>& oPtr) const {
    typename boost::mpl::if_c<has_setPtr<T>::value,
    DoSetPtr<T>,
    DoNotSetPtr<T> >::type maybe_filler;
    maybe_filler(this->obj, iToType, iIndices, oPtr);
  }

  // This is an attempt to optimize for speed, by avoiding the copying
  // of large objects of type T. In this initial version, we assume
  // that for any class having a 'swap' member function should call
  // 'swap' rather than copying the object.

  template <typename T>
  struct DoSwap {
    void operator()(T& a, T& b) { a.swap(b); }
  };

  template <typename T>
  struct DoAssign {
    void operator()(T& a, T& b) { a = b; }
  };

#ifndef __GCCXML__
  template <typename T>
  struct IsMergeable {
    bool operator()(T const&) const { return true; }
  };

  template <typename T>
  struct IsNotMergeable {
    bool operator()(T const&) const { return false; }
  };

  template <typename T>
  struct DoMergeProduct {
    bool operator()(T& a, T const& b) { return a.mergeProduct(b); }
  };

  template <typename T>
  struct DoNotMergeProduct {
    bool operator()(T&, T const&) { return true; }
  };

  template <typename T>
  struct DoHasIsProductEqual {
    bool operator()(T const&) const { return true; }
  };

  template <typename T>
  struct DoNotHasIsProductEqual {
    bool operator()(T const&) const { return false; }
  };

  template <typename T>
  struct DoIsProductEqual {
    bool operator()(T const& a, T const& b) const { return a.isProductEqual(b); }
  };

  template <typename T>
  struct DoNotIsProductEqual {
    bool operator()(T const&, T const&) const { return true; }
  };
#endif

  //------------------------------------------------------------
  // Metafunction support for compile-time selection of code used in
  // Wrapper constructor
  //

  namespace detail {
    typedef char (& no_tag)[1]; // type indicating FALSE
    typedef char (& yes_tag)[2]; // type indicating TRUE

    // Definitions for the following struct and function templates are
    // not needed; we only require the declarations.
    template <typename T, void (T::*)(T&)>  struct swap_function;
    template <typename T> no_tag  has_swap_helper(...);
    template <typename T> yes_tag has_swap_helper(swap_function<T, &T::swap> * dummy);

    template<typename T>
    struct has_swap_function {
      static bool const value =
        sizeof(has_swap_helper<T>(0)) == sizeof(yes_tag);
    };

#ifndef __GCCXML__
    template <typename T, bool (T::*)(T const&)>  struct mergeProduct_function;
    template <typename T> no_tag  has_mergeProduct_helper(...);
    template <typename T> yes_tag has_mergeProduct_helper(mergeProduct_function<T, &T::mergeProduct> * dummy);

    template<typename T>
    struct has_mergeProduct_function {
      static bool const value =
        sizeof(has_mergeProduct_helper<T>(0)) == sizeof(yes_tag);
    };

    template <typename T, bool (T::*)(T const&) const> struct isProductEqual_function;
    template <typename T> no_tag  has_isProductEqual_helper(...);
    template <typename T> yes_tag has_isProductEqual_helper(isProductEqual_function<T, &T::isProductEqual> * dummy);

    template<typename T>
    struct has_isProductEqual_function {
      static bool const value =
        sizeof(has_isProductEqual_helper<T>(0)) == sizeof(yes_tag);
    };
#endif
  }

  template <typename T>
  Wrapper<T>::Wrapper(std::auto_ptr<T> ptr) :
    EDProduct(),
    present(ptr.get() != 0),
    obj() {
    if (present) {
      // The following will call swap if T has such a function,
      // and use assignment if T has no such function.
      typename boost::mpl::if_c<detail::has_swap_function<T>::value,
        DoSwap<T>,
        DoAssign<T> >::type swap_or_assign;
      swap_or_assign(obj, *ptr);
    }
  }

  template <typename T>
  Wrapper<T>::Wrapper(T* ptr) :
  EDProduct(),
  present(ptr != 0),
  obj() {
     std::auto_ptr<T> temp(ptr);
     if (present) {
        // The following will call swap if T has such a function,
        // and use assignment if T has no such function.
        typename boost::mpl::if_c<detail::has_swap_function<T>::value,
        DoSwap<T>,
        DoAssign<T> >::type swap_or_assign;
        swap_or_assign(obj, *ptr);
     }

  }

#ifndef __GCCXML__
  template <typename T>
  bool Wrapper<T>::isMergeable_() const {
    typename boost::mpl::if_c<detail::has_mergeProduct_function<T>::value,
      IsMergeable<T>,
      IsNotMergeable<T> >::type is_mergeable;
    return is_mergeable(obj);
  }

  template <typename T>
  bool Wrapper<T>::mergeProduct_(EDProduct const* newProduct) {
    Wrapper<T> const* wrappedNewProduct = dynamic_cast<Wrapper<T> const*>(newProduct);
    assert(wrappedNewProduct != nullptr);
    typename boost::mpl::if_c<detail::has_mergeProduct_function<T>::value,
      DoMergeProduct<T>,
      DoNotMergeProduct<T> >::type merge_product;
    return merge_product(obj, wrappedNewProduct->obj);
  }

  template <typename T>
  bool Wrapper<T>::hasIsProductEqual_() const {
    typename boost::mpl::if_c<detail::has_isProductEqual_function<T>::value,
      DoHasIsProductEqual<T>,
      DoNotHasIsProductEqual<T> >::type has_is_equal;
    return has_is_equal(obj);
  }

  template <typename T>
  bool Wrapper<T>::isProductEqual_(EDProduct const* newProduct) const {
    Wrapper<T> const* wrappedNewProduct = dynamic_cast<Wrapper<T> const*>(newProduct);
    assert(wrappedNewProduct != nullptr);
    typename boost::mpl::if_c<detail::has_isProductEqual_function<T>::value,
      DoIsProductEqual<T>,
      DoNotIsProductEqual<T> >::type is_equal;
    return is_equal(obj, wrappedNewProduct->obj);
  }
#endif
}

#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

namespace edm {
  namespace helpers {
    template<typename T>
    struct ViewFiller {
      static void fill(T const& obj,
                       ProductID const& id,
                       std::vector<void const*>& pointers,
                       helper_vector_ptr & helpers) {
        /// the following shoudl work also if T is a RefVector<C>
        typedef Ref<T> ref;
        typedef RefVector<T, typename ref::value_type, typename ref::finder_type> ref_vector;
        helpers = helper_vector_ptr(new reftobase::RefVectorHolder<ref_vector>);
        // fillView is the name of an overload set; each concrete
        // collection T should supply a fillView function, in the same
        // namespace at that in which T is defined, or in the 'edm'
        // namespace.
        fillView(obj, id, pointers, * helpers);
        assert(pointers.size() == helpers->size());
     }
    };

    template<typename T>
    struct ViewFiller<RefToBaseVector<T> > {
      static void fill(RefToBaseVector<T> const& obj,
                       ProductID const&,
                       std::vector<void const*>& pointers,
                       helper_vector_ptr & helpers) {
        std::auto_ptr<helper_vector> h = obj.vectorHolder();
        if(h.get() != 0) {
          pointers.reserve(h->size());
          // NOTE: the following implementation has unusual signature!
          fillView(obj, pointers);
          helpers = helper_vector_ptr(h.release());
        }
      }
    };

    template<typename T>
    struct ViewFiller<PtrVector<T> > {
      static void fill(PtrVector<T> const& obj,
                       ProductID const&,
                       std::vector<void const*>& pointers,
                       helper_vector_ptr & helpers) {
        std::auto_ptr<helper_vector> h(new reftobase::RefVectorHolder<PtrVector<T> >(obj));
        if(h.get() != 0) {
          pointers.reserve(obj.size());
          // NOTE: the following implementation has unusual signature!
          fillView(obj, pointers);
          helpers = helper_vector_ptr(h.release());
        }
      }
    };

    template<typename T>
      struct PtrSetter {
        static void set(T const& obj,
                         std::type_info const& iToType,
                         unsigned long iIndex,
                         void const*& oPtr) {
          // setPtr is the name of an overload set; each concrete
          // collection T should supply a fillView function, in the same
          // namespace at that in which T is defined, or in the 'edm'
          // namespace.
          setPtr(obj, iToType, iIndex, oPtr);
        }

        static void fill(T const& obj,
                         std::type_info const& iToType,
                         std::vector<unsigned long> const& iIndex,
                         std::vector<void const*>& oPtr) {
          // fillPtrVector is the name of an overload set; each concrete
          // collection T should supply a fillPtrVector function, in the same
          // namespace at that in which T is defined, or in the 'edm'
          // namespace.
          fillPtrVector(obj, iToType, iIndex, oPtr);
        }
      };
  }

#ifndef __GCCXML__
  template <typename T>
  void DoFillView<T>::operator()(T const& obj,
                                 ProductID const& id,
                                 std::vector<void const*>& pointers,
                                 helper_vector_ptr& helpers) const {
    helpers::ViewFiller<T>::fill(obj, id, pointers, helpers);
  }
#endif

  template <typename T>
  void DoSetPtr<T>::operator()(T const& obj,
                               std::type_info const& iToType,
                               unsigned long iIndex,
                               void const*& oPtr) const {
    helpers::PtrSetter<T>::set(obj, iToType, iIndex, oPtr);
  }

  template <typename T>
  void DoSetPtr<T>::operator()(T const& obj,
                               std::type_info const& iToType,
                               std::vector<unsigned long> const& iIndices,
                               std::vector<void const*>& oPtr) const {
    helpers::PtrSetter<T>::fill(obj, iToType, iIndices, oPtr);
  }

}

#include "DataFormats/Common/interface/FillView.h"
#include "DataFormats/Common/interface/setPtr.h"
#include "DataFormats/Common/interface/fillPtrVector.h"

#endif
