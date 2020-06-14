#include <tuple>
#include <type_traits>

template <class Func, class... Args>
class LazyResult {
public:
  using Result = typename std::invoke_result<Func, Args...>::type;

  // For the Args... we explicitly add const& (also in the the args_ tuple).
  // Otherwise, the arguments will be stored by value which comes with too
  // much overhead.  This implies that the lifetime of the arguments passed
  // to Lazy neet to live longer than the Lazy instance. Function pointers
  // are small, so no need for const& to the Func.
  LazyResult(Func func, Args const&... args) : func_(func), args_(args...) {}

  // Get the actual result. An alternative to using a ::value() member
  // function could be a cast operator: operator Result const &().
  // This might be pretty because the result is automatically evaluated the
  // first time you try to bind it to a Result const &. I think this is
  // however a bit too implicit and dangerous.
  Result const& value() {
    if (!evaluated_) {
      result_ = func_(std::get<Args const&>(args_)...);
      evaluated_ = true;
    }
    return result_;
  }

private:
  // The evaluated_ and func_ member might both be just one byte (if the
  // fuction is a lambda function for example), so by having them together we
  // save alignment bits if this is the case.
  bool evaluated_ = false;
  Func func_;
  std::tuple<Args const&...> args_;
  Result result_;
};
