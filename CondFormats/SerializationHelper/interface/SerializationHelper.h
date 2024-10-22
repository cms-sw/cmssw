#ifndef CondFormats_SerializationHelper_SerializationHelper_h
#define CondFormats_SerializationHelper_SerializationHelper_h
// -*- C++ -*-
//
// Package:     CondFormats/SerializationHelper
// Class  :     SerializationHelper
//
/**\class SerializationHelper SerializationHelper.h "CondFormats/SerializationHelper/interface/SerializationHelper.h"

 Description: concrete implementation of the SerializationHelperBase interface

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 31 May 2023 14:55:13 GMT
//

// system include files
#include <string_view>

// user include files
#include "CondFormats/SerializationHelper/interface/SerializationHelperBase.h"
#include "CondFormats/Serialization/interface/Archive.h"
#include "FWCore/Utilities/interface/mplVector.h"

// forward declarations

namespace cond::serialization {
  struct NoInitializer {};

  template <typename T>
  struct ClassName;

  template <typename T>
  std::unique_ptr<T> makeClass() {
    return std::make_unique<T>();
  }

  template <typename T>
  struct BaseClassInfo {
    constexpr static bool kAbstract = false;
    using inheriting_classes_t = edm::mpl::Vector<>;
  };

  template <typename T, bool IsAbstract, typename... INHERITING>
  struct BaseClassInfoImpl {
    constexpr static bool kAbstract = IsAbstract;
    using inheriting_classes_t = edm::mpl::Vector<INHERITING...>;
  };

  template <typename T, typename INIT = NoInitializer>
  class SerializationHelper : public SerializationHelperBase {
  public:
    SerializationHelper() = default;

    SerializationHelper(const SerializationHelper&) = delete;                   // stop default
    const SerializationHelper& operator=(const SerializationHelper&) = delete;  // stop default

    // ---------- const member functions ---------------------

    unique_void_ptr deserialize(std::streambuf& iBuff, const std::string_view iClassName) const final {
      if constexpr (not BaseClassInfo<T>::kAbstract) {
        using BaseClassAndInheriting =
            typename edm::mpl::Push<T, typename BaseClassInfo<T>::inheriting_classes_t>::Result;
        return deserialize_impl<BaseClassAndInheriting>(iBuff, iClassName);
      } else {
        return deserialize_impl<typename BaseClassInfo<T>::inheriting_classes_t>(iBuff, iClassName);
      }
    }

    std::string_view serialize(std::streambuf& oBuff, void const* iObj) const final {
      auto iTypedObjectPtr = static_cast<T const*>(iObj);
      if constexpr (BaseClassInfo<T>::kAbstract) {
        return serialize_impl<typename BaseClassInfo<T>::inheriting_classes_t>(oBuff, iTypedObjectPtr);
      } else {
        using BaseClassAndInheriting =
            typename edm::mpl::Push<T, typename BaseClassInfo<T>::inheriting_classes_t>::Result;
        return serialize_impl<BaseClassAndInheriting>(oBuff, iTypedObjectPtr);
      }
    }

    const std::type_info& type() const final { return typeid(T); }

  private:
    template <typename TYPELIST>
    static unique_void_ptr deserialize_impl(std::streambuf& iBuff, const std::string_view iClassName) {
      if constexpr (edm::mpl::Pop<TYPELIST>::empty) {
        return {};
      } else {
        using CheckType = typename edm::mpl::Pop<TYPELIST>::Item;
        if (iClassName == ClassName<CheckType>::kName) {
          std::unique_ptr<CheckType> tmp = makeClass<CheckType>();

          InputArchive ia(iBuff);
          ia >> (*tmp);
          if constexpr (not std::is_same_v<INIT, NoInitializer>) {
            INIT init;
            init(*tmp);
          }
          return unique_void_ptr(tmp.release(), [](const void* iPtr) { delete static_cast<const T*>(iPtr); });

        } else {
          return deserialize_impl<typename edm::mpl::Pop<TYPELIST>::Remaining>(iBuff, iClassName);
        }
      }
    }
    template <typename TYPELIST>
    static std::string_view serialize_impl(std::streambuf& oBuff, T const* iObj) {
      if constexpr (edm::mpl::Pop<TYPELIST>::empty) {
        return {};
      } else {
        if (typeid(*iObj) == typeid(typename edm::mpl::Pop<TYPELIST>::Item)) {
          auto iTypedObjectPtr = dynamic_cast<typename edm::mpl::Pop<TYPELIST>::Item const*>(iObj);
          OutputArchive oa(oBuff);
          oa << *iTypedObjectPtr;

          return ClassName<typename edm::mpl::Pop<TYPELIST>::Item>::kName;

        } else {
          return serialize_impl<typename edm::mpl::Pop<TYPELIST>::Remaining>(oBuff, iObj);
        }
      }
    }
  };
}  // namespace cond::serialization
#endif
