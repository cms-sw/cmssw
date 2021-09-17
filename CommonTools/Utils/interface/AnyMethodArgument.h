#ifndef CommonTools_Utils_AnyMethodArgument_h
#define CommonTools_Utils_AnyMethodArgument_h

#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "CommonTools/Utils/interface/Exception.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>

namespace reco {
  namespace parser {

    // true if T matches one of the integral types in the default
    // AnyMethodArgument variant
    template <typename T>
    struct matches_another_integral_type {
      static bool const value = std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
                                std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
                                std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
                                std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value;
    };

    // size_t on 32-bit Os X is type unsigned long, which doesn't match uint32_t,
    // so add unsigned long if it doesn't match any of the other integral types.
    // Use "unsigned long" rather than size_t as PtrVector has unsigned long as
    // size_type
    typedef std::conditional<
        matches_another_integral_type<unsigned long>::value,
        std::variant<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, double, float, std::string>,
        std::variant<int8_t,
                     uint8_t,
                     int16_t,
                     uint16_t,
                     int32_t,
                     uint32_t,
                     int64_t,
                     uint64_t,
                     unsigned long,
                     double,
                     float,
                     std::string> >::type AnyMethodArgument;

    class AnyMethodArgumentFixup {
    private:
      edm::TypeWithDict dataType_;
      template <typename From, typename To>
      std::pair<AnyMethodArgument, int> retOk_(const From &f, int cast) const {
        return std::pair<AnyMethodArgument, int>(AnyMethodArgument(static_cast<To>(f)), cast);
      }

      // correct return for each int output type
      std::pair<AnyMethodArgument, int> doInt(int t) const {
        if (dataType_ == typeid(int8_t)) {
          return retOk_<int, int8_t>(t, 0);
        }
        if (dataType_ == typeid(uint8_t)) {
          return retOk_<int, uint8_t>(t, 0);
        }
        if (dataType_ == typeid(int16_t)) {
          return retOk_<int, int16_t>(t, 0);
        }
        if (dataType_ == typeid(uint16_t)) {
          return retOk_<int, uint16_t>(t, 0);
        }
        if (dataType_ == typeid(int32_t)) {
          return retOk_<int, int32_t>(t, 0);
        }
        if (dataType_ == typeid(uint32_t)) {
          return retOk_<int, uint32_t>(t, 0);
        }
        if (dataType_ == typeid(int64_t)) {
          return retOk_<int, int64_t>(t, 0);
        }
        if (dataType_ == typeid(uint64_t)) {
          return retOk_<int, uint64_t>(t, 0);
        }
        if (dataType_ == typeid(unsigned long)) {
          return retOk_<int, unsigned long>(t, 0);
        }  // harmless if unsigned long matches another type
        if (dataType_ == typeid(double)) {
          return retOk_<int, double>(t, 1);
        }
        if (dataType_ == typeid(float)) {
          return retOk_<int, float>(t, 1);
        }
        return std::pair<AnyMethodArgument, int>(t, -1);
      }

    public:
      AnyMethodArgumentFixup(const edm::TypeWithDict &type) : dataType_(type) {}

      // we handle all integer types through 'int', as that's the way they are parsed by boost::spirit
      template <typename I>
      typename std::enable_if<std::is_integral<I>::value, std::pair<AnyMethodArgument, int> >::type operator()(
          const I &t) const {
        return doInt(t);
      }

      template <typename F>
      typename std::enable_if<std::is_floating_point<F>::value, std::pair<AnyMethodArgument, int> >::type operator()(
          const F &t) const {
        if (dataType_ == typeid(double)) {
          return retOk_<F, double>(t, 0);
        }
        if (dataType_ == typeid(float)) {
          return retOk_<F, float>(t, 0);
        }
        return std::pair<AnyMethodArgument, int>(t, -1);
      }

      std::pair<AnyMethodArgument, int> operator()(const std::string &t) const {
        if (dataType_.isEnum()) {
          if (dataType_.dataMemberSize() == 0) {
            throw parser::Exception(t.c_str())
                << "Enumerator '" << dataType_.name() << "' has no keys.\nPerhaps the dictionary is missing?\n";
          }
          int ival = dataType_.stringToEnumValue(t);
          // std::cerr << "  value is = " << dataType_.stringToEnumValue(t) << std::endl;
          return std::pair<AnyMethodArgument, int>(ival, 1);
        }
        if (dataType_ == typeid(std::string)) {
          return std::pair<AnyMethodArgument, int>(t, 0);
        }
        return std::pair<AnyMethodArgument, int>(t, -1);
      }
    };

    class AnyMethodArgument2VoidPtr {
    public:
      template <typename T>
      void *operator()(const T &t) const {
        return const_cast<void *>(static_cast<const void *>(&t));
      }
    };
  }  // namespace parser
}  // namespace reco

#endif
