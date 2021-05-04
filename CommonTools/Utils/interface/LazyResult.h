#ifndef CommonTools_Utils_LazyResult_h
#define CommonTools_Utils_LazyResult_h
// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     LazyResult
//
/**\class LazyResult LazyResult.h "CommonTools/Utils/interface/LazyResult.h"
 Description: Wrapper around a function call for lazy execution.

 Usage:
    // example: lazy addition
    auto result = LazyResult(std::plus<int>, x, x);
    std::cout << result.value() << std::endl;

Notes:
  * The arguments for the delayed call are stored by reference (watch their
    lifetime).
  * The overhead in memory compared to just storing the result is small: one
    reference per argument, one bool flag and a function pointer (on my system:
    1 byte for lambda function, 8 bytes for global function and 16 bytes for
    member function due to possible index to virtual table).

Implementation:

  * For the Args... we explicitly add const& (also in the the args_ tuple).
    Otherwise, the arguments will be stored by value which comes with too much
    overhead. This implies that the lifetime of the arguments passed to
    LazyResult neet to live longer than the LazyResult instance. Function pointers
    are small, so no need for const& to the Func.
  * An alternative to using a ::value() member function to get the result could
    be a cast operator: operator Result const &(). This might be pretty because
    the result is automatically evaluated the first time you try to bind it to
    a Result const &. I think this would however be too implicit and dangerous.

*/
//
// Original Author:  Jonas Rembser
//         Created:  Mon, 14 Jun 2020 00:00:00 GMT
//

#include <tuple>
#include <type_traits>

template <class Func, class... Args>
class LazyResult {
public:
  using Result = typename std::invoke_result<Func, Args...>::type;

  LazyResult(Func func, Args const&... args) : func_(func), args_(args...) {}

  Result const& value() {
    if (!evaluated_) {
      evaluate();
    }
    return result_;
  }

private:
  void evaluate() { evaluateImpl(std::make_index_sequence<sizeof...(Args)>{}); }

  template <std::size_t... ArgIndices>
  void evaluateImpl(std::index_sequence<ArgIndices...>) {
    result_ = func_(std::get<ArgIndices>(args_)...);
    evaluated_ = true;
  }

  // having evaluated_ and the potentially small func_ together might
  // save alignemnt bits (e.g. a lambda function is just 1 byte)
  bool evaluated_ = false;
  Func func_;
  std::tuple<Args const&...> args_;
  Result result_;
};

#endif
