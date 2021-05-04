#ifndef CommonTools_Utils_LazyConstructed_h
#define CommonTools_Utils_LazyConstructed_h
// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     LazyConstructed
//
/**\class LazyConstructed LazyConstructed.h "CommonTools/Utils/interface/LazyConstructed.h"
 Description: Wrapper around a class for lazy construction.

 Usage:
    // example: lazy SoA table
    auto object = makeLazy<edm::soa::EtaPhiTable>(trackCollection);

Notes:
  * See similar class CommonTools/Utils/interface/LazyResult.h for implementation details.

*/
//
// Original Author:  Jonas Rembser
//         Created:  Mon, 14 Aug 2020 16:05:45 GMT
//
//
#include <tuple>
#include <optional>

template <class WrappedClass, class... Args>
class LazyConstructed {
public:
  LazyConstructed(Args const&... args) : args_(args...) {}

  WrappedClass& value() {
    if (!object_) {
      evaluate();
    }
    return object_.value();
  }

private:
  void evaluate() { evaluateImpl(std::make_index_sequence<sizeof...(Args)>{}); }

  template <std::size_t... ArgIndices>
  void evaluateImpl(std::index_sequence<ArgIndices...>) {
    object_ = WrappedClass(std::get<ArgIndices>(args_)...);
  }

  std::optional<WrappedClass> object_ = std::nullopt;
  std::tuple<Args const&...> args_;
};

// helper function to create a LazyConstructed where the Args are deduced from the function argument types
template <class WrappedClass, class... Args>
auto makeLazy(Args&&... args) {
  return LazyConstructed<WrappedClass, Args...>(std::forward<Args>(args)...);
}

#endif
