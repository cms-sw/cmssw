#ifndef DataFormats_Common_WrapperDetail_h
#define DataFormats_Common_WrapperDetail_h

/*----------------------------------------------------------------------

WrapperDetail: Metafunction support for compile-time selection of code.

----------------------------------------------------------------------*/

#include <typeinfo>
namespace edm {

  //Need to specialize the case of std::vector<edm::Ptr<T>>
  template<typename T> class Ptr;

  namespace detail {
    typedef char (& no_tag)[1]; // type indicating FALSE
    typedef char (& yes_tag)[2]; // type indicating TRUE

    // void swap_or_assign(T& a, T& b) will swap if T::swap(T&) is defined and assign otherwise
    // Definitions for the following struct and function templates are not needed; we only require the declarations.
    template<typename T, void (T::*)(T&)> struct swap_function;
    template<typename T> static yes_tag has_swap(swap_function<T, &T::swap>* dummy);
    template<typename T> static no_tag has_swap(...);

    template<typename T>
    struct has_swap_function {
      static bool const value = sizeof(has_swap<T>(0)) == sizeof(yes_tag);
    };

    template<typename T, bool = has_swap_function<T>::value> struct doSwapOrAssign;
    template<typename T> struct doSwapOrAssign<T, true> {
      void operator()(T& thisProduct, T& otherProduct) {
        thisProduct.swap(otherProduct);
      }
    };
    template<typename T> struct doSwapOrAssign<T, false> {
      void operator()(T& thisProduct, T& otherProduct) {
        thisProduct = otherProduct;
      }
    };

#ifndef __GCCXML__
    // valueTypeInfo_() will return typeid(T::value_type) if T::value_type is declared and typeid(void) otherwise.
    // Definitions for the following struct and function templates are not needed; we only require the declarations.
    template<typename T> static yes_tag& has_value_type(typename T::value_type*);
    template<typename T> static no_tag& has_value_type(...);

    template<typename T> struct has_typedef_value_type {
      static const bool value = sizeof(has_value_type<T>(nullptr)) == sizeof(yes_tag);
    };
    template<typename T, bool = has_typedef_value_type<T>::value> struct getValueType;
    template<typename T> struct getValueType<T, true> {
      std::type_info const& operator()() {
        return typeid(typename T::value_type);
      }
    };
    template<typename T> struct getValueType<T, false> {
      std::type_info const& operator()() {
        return typeid(void);
      }
    };

    // memberTypeInfo_() will return typeid(T::member_type) if T::member_type is declared and typeid(void) otherwise.
    // Definitions for the following struct and function templates are not needed; we only require the declarations.
    template<typename T> static yes_tag& has_member_type(typename T::member_type*);
    template<typename T> static no_tag& has_member_type(...);

    template<typename T> struct has_typedef_member_type {
      static const bool value = sizeof(has_member_type<T>(nullptr)) == sizeof(yes_tag);
    };
    template<typename T, bool = has_typedef_member_type<T>::value> struct getMemberType;
    template<typename T> struct getMemberType<T, true> {
      std::type_info const& operator()() {
        return typeid(typename T::member_type);
      }
    };
    template<typename T> struct getMemberType<T, false> {
      std::type_info const& operator()() {
        return typeid(void);
      }
    };

    template< typename T> struct has_typedef_member_type<std::vector<edm::Ptr<T> > > {
      static const bool value = true;
    };

    template <typename T> struct getMemberType<std::vector<edm::Ptr<T> >, true> {
      std::type_info const& operator()() {
        return typeid(T);
      }
    };

    // bool isMergeable_() will return true if T::mergeProduct(T const&) is declared and false otherwise
    // bool mergeProduct_(WrapperBase const*) will merge products if T::mergeProduct(T const&) is defined
    // Definitions for the following struct and function templates are not needed; we only require the declarations.
    template<typename T, bool (T::*)(T const&)> struct mergeProduct_function;
    template<typename T> static yes_tag has_mergeProduct(mergeProduct_function<T, &T::mergeProduct>* dummy);
    template<typename T> static no_tag has_mergeProduct(...);

    template<typename T>
    struct has_mergeProduct_function {
      static bool const value =
        sizeof(has_mergeProduct<T>(0)) == sizeof(yes_tag);
    };

    template<typename T, bool = has_mergeProduct_function<T>::value> struct getHasMergeFunction;
    template<typename T> struct getHasMergeFunction<T, true> {
      bool operator()() {
        return true;
      }
    };
    template<typename T> struct getHasMergeFunction<T, false> {
      bool operator()() {
        return false;
      }
    };
    template<typename T, bool = has_mergeProduct_function<T>::value> struct doMergeProduct;
    template<typename T> struct doMergeProduct<T, true> {
      bool operator()(T& thisProduct, T const& newProduct) {
        return thisProduct.mergeProduct(newProduct);
      }
    };
    template<typename T> struct doMergeProduct<T, false> {
      bool operator()(T& thisProduct, T const& newProduct) {
        return true; // Should never be called
      }
    };

    // bool hasIsProductEqual_() will return true if T::isProductEqual(T const&) const is declared and false otherwise
    // bool isProductEqual _(WrapperBase const*) will call T::isProductEqual(T const&) if it is defined
    // Definitions for the following struct and function templates are not needed; we only require the declarations.
    template<typename T, bool (T::*)(T const&) const> struct isProductEqual_function;
    template<typename T> static yes_tag has_isProductEqual(isProductEqual_function<T, &T::isProductEqual>* dummy);
    template<typename T> static no_tag has_isProductEqual(...);

    template<typename T>
    struct has_isProductEqual_function {
      static bool const value =
        sizeof(has_isProductEqual<T>(0)) == sizeof(yes_tag);
    };

    template<typename T, bool = has_isProductEqual_function<T>::value> struct getHasIsProductEqual;
    template<typename T> struct getHasIsProductEqual<T, true> {
      bool operator()() {
        return true;
      }
    };
    template<typename T> struct getHasIsProductEqual<T, false> {
      bool operator()() {
        return false;
      }
    };
    template<typename T, bool = has_isProductEqual_function<T>::value> struct doIsProductEqual;
    template<typename T> struct doIsProductEqual<T, true> {
      bool operator()(T const& thisProduct, T const& newProduct) {
        return thisProduct.isProductEqual(newProduct);
      }
    };
    template<typename T> struct doIsProductEqual<T, false> {
      bool operator()(T const& thisProduct, T const& newProduct) {
        return true; // Should never be called
      }
    };
#endif
  }
}
#endif
