#ifndef FWCore_Framework_interface_es_impl_ReturnArgumentTypes_h
#define FWCore_Framework_interface_es_impl_ReturnArgumentTypes_h

namespace edm {
  class WaitingTaskWithArenaHolder;

  namespace eventsetup::impl {
    template <typename F>
    struct ReturnArgumentTypesImpl;

    // function pointer
    template <typename R, typename T>
    struct ReturnArgumentTypesImpl<R (*)(T const&)> {
      using argument_type = T;
      using return_type = R;
    };
    template <typename R, typename T>
    struct ReturnArgumentTypesImpl<R (*)(T const&, WaitingTaskWithArenaHolder)> {
      using argument_type = T;
      using return_type = R;
    };

    // mutable functor/lambda
    template <typename R, typename T, typename O>
    struct ReturnArgumentTypesImpl<R (O::*)(T const&)> {
      using argument_type = T;
      using return_type = R;
    };
    template <typename R, typename T, typename O>
    struct ReturnArgumentTypesImpl<R (O::*)(T const&, WaitingTaskWithArenaHolder)> {
      using argument_type = T;
      using return_type = R;
    };

    // const functor/lambda
    template <typename R, typename T, typename O>
    struct ReturnArgumentTypesImpl<R (O::*)(T const&) const> {
      using argument_type = T;
      using return_type = R;
    };
    template <typename R, typename T, typename O>
    struct ReturnArgumentTypesImpl<R (O::*)(T const&, WaitingTaskWithArenaHolder) const> {
      using argument_type = T;
      using return_type = R;
    };

    //////////

    template <typename F>
    struct ReturnArgumentTypes;

    template <typename F>
      requires std::is_class_v<F>
    struct ReturnArgumentTypes<F> {
      using argument_type = typename ReturnArgumentTypesImpl<decltype(&F::operator())>::argument_type;
      using return_type = typename ReturnArgumentTypesImpl<decltype(&F::operator())>::return_type;
    };

    template <typename F>
      requires std::is_pointer_v<F>
    struct ReturnArgumentTypes<F> {
      using argument_type = typename ReturnArgumentTypesImpl<F>::argument_type;
      using return_type = typename ReturnArgumentTypesImpl<F>::return_type;
    };

  }  // namespace eventsetup::impl
}  // namespace edm

#endif
