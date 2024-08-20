#ifndef DataFormats_Common_WrapperDetail_h
#define DataFormats_Common_WrapperDetail_h

/*----------------------------------------------------------------------

WrapperDetail: Metafunction support for compile-time selection of code.

----------------------------------------------------------------------*/

#include <memory>
#include <typeinfo>
#include <type_traits>
#include <vector>

namespace edm {

  //Need to specialize the case of std::vector<edm::Ptr<T>>
  template <typename T>
  class Ptr;

  namespace detail {
    using no_tag = std::false_type;  // type indicating FALSE
    using yes_tag = std::true_type;  // type indicating TRUE

    // valueTypeInfo_() will return typeid(T::value_type) if T::value_type is declared and typeid(void) otherwise.
    // Definitions for the following struct and function templates are not needed; we only require the declarations.
    template <typename T>
    static yes_tag has_value_type(typename T::value_type*);
    template <typename T>
    static no_tag has_value_type(...);

    template <typename T>
    struct has_typedef_value_type {
      static constexpr bool value = std::is_same<decltype(has_value_type<T>(nullptr)), yes_tag>::value;
    };
    template <typename T, bool = has_typedef_value_type<T>::value>
    struct getValueType;
    template <typename T>
    struct getValueType<T, true> {
      static constexpr std::type_info const& get() { return typeid(typename T::value_type); }
    };
    template <typename T>
    struct getValueType<T, false> {
      static constexpr std::type_info const& get() { return typeid(void); }
    };

    // memberTypeInfo_() will return typeid(T::member_type) if T::member_type is declared and typeid(void) otherwise.
    // Definitions for the following struct and function templates are not needed; we only require the declarations.
    template <typename T>
    static yes_tag has_member_type(typename T::member_type*);
    template <typename T>
    static no_tag has_member_type(...);

    template <typename T>
    struct has_typedef_member_type {
      static constexpr bool value = std::is_same<decltype(has_member_type<T>(nullptr)), yes_tag>::value;
    };
    template <typename T, bool = has_typedef_member_type<T>::value>
    struct getMemberType;
    template <typename T>
    struct getMemberType<T, true> {
      static constexpr std::type_info const& get() { return typeid(typename T::member_type); }
    };
    template <typename T>
    struct getMemberType<T, false> {
      static constexpr std::type_info const& get() { return typeid(void); }
    };

    template <typename T>
    struct has_typedef_member_type<std::vector<edm::Ptr<T> > > {
      static constexpr bool value = true;
    };

    template <typename T>
    struct getMemberType<std::vector<edm::Ptr<T> >, true> {
      static constexpr std::type_info const& get() { return typeid(T); }
    };

    template <typename T, typename Deleter>
    struct has_typedef_member_type<std::vector<std::unique_ptr<T, Deleter> > > {
      static constexpr bool value = true;
    };

    template <typename T, typename Deleter>
    struct getMemberType<std::vector<std::unique_ptr<T, Deleter> >, true> {
      static constexpr std::type_info const& get() { return typeid(T); }
    };

    // bool isMergeable_() will return true if T::mergeProduct(T const&) is declared and false otherwise
    // bool mergeProduct_(WrapperBase const*) will merge products if T::mergeProduct(T const&) is defined
    // Definitions for the following struct and function templates are not needed; we only require the declarations.
    template <typename T, bool (T::*)(T const&)>
    struct mergeProduct_function;
    template <typename T>
    static yes_tag has_mergeProduct(mergeProduct_function<T, &T::mergeProduct>* dummy);
    template <typename T>
    static no_tag has_mergeProduct(...);

    template <typename T>
    struct has_mergeProduct_function {
      static constexpr bool value = std::is_same<decltype(has_mergeProduct<T>(nullptr)), yes_tag>::value;
    };

    template <typename T>
    inline constexpr bool has_mergeProduct_function_v = has_mergeProduct_function<T>::value;

    // bool hasIsProductEqual_() will return true if T::isProductEqual(T const&) const is declared and false otherwise
    // bool isProductEqual _(WrapperBase const*) will call T::isProductEqual(T const&) if it is defined
    // Definitions for the following struct and function templates are not needed; we only require the declarations.
    template <typename T, bool (T::*)(T const&) const>
    struct isProductEqual_function;
    template <typename T>
    static yes_tag has_isProductEqual(isProductEqual_function<T, &T::isProductEqual>* dummy);
    template <typename T>
    static no_tag has_isProductEqual(...);

    template <typename T>
    struct has_isProductEqual_function {
      static constexpr bool value = std::is_same<decltype(has_isProductEqual<T>(nullptr)), yes_tag>::value;
    };
    template <typename T>
    inline constexpr bool has_isProductEqual_function_v = has_isProductEqual_function<T>::value;

    // bool hasSwap_() will return true if T::swap(T&) is declared and false otherwise
    // void swapProduct_() will call T::swap(T&) if it is defined otherwise it does nothing
    // Definitions for the following struct and function templates are not needed; we only require the declarations.
    template <typename T, void (T::*)(T&)>
    struct swap_function;
    template <typename T>
    static yes_tag has_swap(swap_function<T, &T::swap>* dummy);
    template <typename T>
    static no_tag has_swap(...);

    template <typename T>
    struct has_swap_function {
      static constexpr bool value = std::is_same<decltype(has_swap<T>(nullptr)), yes_tag>::value;
    };

    template <typename T>
    inline constexpr bool has_swap_function_v = has_swap_function<T>::value;

  }  // namespace detail
}  // namespace edm
#endif
