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
      std::type_info const& operator()() { return typeid(typename T::value_type); }
    };
    template <typename T>
    struct getValueType<T, false> {
      std::type_info const& operator()() { return typeid(void); }
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
      std::type_info const& operator()() { return typeid(typename T::member_type); }
    };
    template <typename T>
    struct getMemberType<T, false> {
      std::type_info const& operator()() { return typeid(void); }
    };

    template <typename T>
    struct has_typedef_member_type<std::vector<edm::Ptr<T> > > {
      static constexpr bool value = true;
    };

    template <typename T>
    struct getMemberType<std::vector<edm::Ptr<T> >, true> {
      std::type_info const& operator()() { return typeid(T); }
    };

    template <typename T, typename Deleter>
    struct has_typedef_member_type<std::vector<std::unique_ptr<T, Deleter> > > {
      static constexpr bool value = true;
    };

    template <typename T, typename Deleter>
    struct getMemberType<std::vector<std::unique_ptr<T, Deleter> >, true> {
      std::type_info const& operator()() { return typeid(T); }
    };

  }  // namespace detail
}  // namespace edm
#endif
