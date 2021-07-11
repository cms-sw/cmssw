#ifndef FWCore_Concurrency_beginChain_h
#define FWCore_Concurrency_beginChain_h
// -*- C++ -*-
//
// Package:     Concurrency
// function  :     beginChain
//
/**\function beginChain

 Description: Handles creation of a chain of WaitingTasks

 Usage:
    Use the function to begin constructing a chain of waiting tasks.
    Once the chain is declared, call end() with a WaitingTaskHolder
    to get back a new WaitingTaskHolder.
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 13:46:31 CST 2013
// $Id$
//

// system include files
#include <atomic>
#include <exception>
#include <memory>
#include <type_traits>

// user include files
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

// forward declarations

namespace edm {
  namespace waiting_task::detail {

    template <class, class = void>
    struct has_exception_handling : std::false_type {};

    template <class T>
    struct has_exception_handling<T,
                                  std::void_t<decltype(std::declval<T&>()(
                                      static_cast<std::exception_ptr const*>(nullptr), edm::WaitingTaskHolder()))>>
        : std::true_type {};

    template <typename F>
    struct AutoExceptionHandler {
      AutoExceptionHandler(F&& iF) : f_{std::forward<F>(iF)} {}

      void operator()(std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) noexcept {
        if (iPtr) {
          h.doneWaiting(*iPtr);
        } else {
          try {
            f_(h);
          } catch (...) {
            h.doneWaiting(std::current_exception());
          }
        }
      }

    private:
      F f_;
    };

    template <typename E, typename F>
    struct ExplicitExceptionHandler {
      ExplicitExceptionHandler(E&& iE, F&& iF) : except_(std::forward<E>(iE)), f_{std::forward<F>(iF)} {}

      void operator()(std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) noexcept {
        if (iPtr) {
          except_(*iPtr);
          h.doneWaiting(*iPtr);
        } else {
          try {
            f_(h);
          } catch (...) {
            h.doneWaiting(std::current_exception());
          }
        }
      }

    private:
      E except_;
      F f_;
    };

    template <typename T>
    struct Conditional {};

    template <typename... T>
    struct WaitingTaskChain;

    template <typename O, typename... T>
    [[nodiscard]] constexpr auto _then(O&& iO, WaitingTaskChain<T...> iChain) {
      if constexpr (has_exception_handling<O>::value) {
        return WaitingTaskChain<O, T...>(std::forward<O>(iO), std::move(iChain));
      } else {
        return WaitingTaskChain<AutoExceptionHandler<O>, T...>(AutoExceptionHandler<O>(std::forward<O>(iO)),
                                                               std::move(iChain));
      }
    }

    template <typename U>
    struct WaitingTaskChain<U> {
      constexpr explicit WaitingTaskChain(U&& iU) : f_{std::forward<U>(iU)} {}

      /**Define next task to run once this task has finished. Two different functor types are allowed
       1. The functor  takes a edm::WaitingTaskHolder as argument. If an exception happened in a previous task, the functor will NOT be run.
       If an exception happens while running the functor, the exception will be propagated to the WaitingTaskHolder.
       2. The functor takes a std::exception_ptr const* and WaitingTaskHolder. If an exception happened in a previous task, the first
       argument will be non-nullptr. In that case, the exception will NOT be automatically propagated to the WaitingTaskHolder. In addition,
       if the functor itself throws an exception, it is up to the functor to handle the exception.
       */
      template <typename O>
      [[nodiscard]] constexpr auto then(O&& iO) {
        return _then(std::forward<O>(iO), std::move(*this));
      }

      ///Define next task to run once this task has finished. If a previous task had an exception, only the first functor, iE, will be run and passed the std::exception_ptr const. If there
      /// was no exception, then only the functor iF will be run. If iF has an exception, it will be automatically propagated to the edm::WaitingTaskHolder.
      template <typename E, typename F>
      [[nodiscard]] constexpr auto thenIfExceptionElse(E&& iE, F&& iF) {
        return WaitingTaskChain<ExplicitExceptionHandler<E, F>, U>(
            ExplicitExceptionHandler<E, F>(std::forward<E>(iE), std::forward<F>(iF)), std::move(*this));
      }

      ///Only runs this task if the condition (which is known at the call time) is true. If false, this task will be skipped and the following task will run.
      template <typename O>
      [[nodiscard]] constexpr auto ifThen(bool iCondition, O&& iO) {
        return WaitingTaskChain<Conditional<AutoExceptionHandler<O>>, U>(
            iCondition, AutoExceptionHandler<O>(std::forward<O>(iO)), std::move(*this));
      }

      [[nodiscard]] WaitingTaskHolder task(WaitingTaskHolder iV) {
        return WaitingTaskHolder(
            *iV.group(),
            make_waiting_task([f = std::move(f_), v = std::move(iV)](const std::exception_ptr* iPtr) mutable {
              f(iPtr, std::move(v));
            }));
      }

      void run(WaitingTaskHolder iH) { f_(nullptr, std::move(iH)); }

    private:
      U f_;
    };

    template <typename U, typename... T>
    struct WaitingTaskChain<U, T...> {
      explicit constexpr WaitingTaskChain(U&& iU, WaitingTaskChain<T...> iL)
          : l_(std::move(iL)), f_{std::forward<U>(iU)} {}

      /**Define next task to run once this task has finished. Two different functor types are allowed
       1. The functor  takes a edm::WaitingTaskHolder as argument. If an exception happened in a previous task, the functor will NOT be run.
       If an exception happens while running the functor, the exception will be propagated to the WaitingTaskHolder.
       2. The functor takes a std::exception_ptr const* and WaitingTaskHolder. If an exception happened in a previous task, the first
       argument will be non-nullptr. In that case, the exception will NOT be automatically propagated to the WaitingTaskHolder. In addition,
       if the functor itself throws an exception, it is up to the functor to handle the exception.
       */
      template <typename O>
      [[nodiscard]] constexpr auto then(O&& iO) {
        return _then(std::forward<O>(iO), std::move(*this));
      }

      ///Define next task to run once this task has finished. If a previous task had an exception, only the first functor, iE, will be run and passed the std::exception_ptr const. If there
      /// was no exception, then only the functor iF will be run. If iF has an exception, it will be automatically propagated to the edm::WaitingTaskHolder.
      template <typename E, typename F>
      [[nodiscard]] constexpr auto thenIfExceptionElse(E&& iE, F&& iF) {
        return WaitingTaskChain<ExplicitExceptionHandler<E, F>, U, T...>(
            ExplicitExceptionHandler<E, F>(std::forward<E>(iE), std::forward<F>(iF)), std::move(*this));
      }

      template <typename O>
      [[nodiscard]] constexpr auto ifThen(bool iCondition, O&& iO) {
        return WaitingTaskChain<Conditional<AutoExceptionHandler<O>>, U, T...>(
            iCondition, AutoExceptionHandler<O>(std::forward<O>(iO)), std::move(*this));
      }

      [[nodiscard]] WaitingTaskHolder task(WaitingTaskHolder iV) {
        return l_.task(WaitingTaskHolder(
            *iV.group(),
            make_waiting_task([f = std::move(f_), v = std::move(iV)](std::exception_ptr const* iPtr) mutable {
              f(iPtr, std::move(v));
            })));
      }

      void run(WaitingTaskHolder iV) {
        l_.run(WaitingTaskHolder(
            *iV.group(),
            make_waiting_task([f = std::move(f_), v = std::move(iV)](std::exception_ptr const* iPtr) mutable {
              f(iPtr, std::move(v));
            })));
      }

    private:
      WaitingTaskChain<T...> l_;
      U f_;
    };

    template <typename U, typename... T>
    struct WaitingTaskChain<Conditional<U>, T...> {
      explicit constexpr WaitingTaskChain(bool iCondition, U&& iU, WaitingTaskChain<T...> iL)
          : l_(std::move(iL)), f_{std::forward<U>(iU)}, condition_(iCondition) {}

      /**Define next task to run once this task has finished. Two different functor types are allowed
       1. The functor  takes a edm::WaitingTaskHolder as argument. If an exception happened in a previous task, the functor will NOT be run.
       If an exception happens while running the functor, the exception will be propagated to the WaitingTaskHolder.
       2. The functor takes a std::exception_ptr const* and WaitingTaskHolder. If an exception happened in a previous task, the first
       argument will be non-nullptr. In that case, the exception will NOT be automatically propagated to the WaitingTaskHolder. In addition,
       if the functor itself throws an exception, it is up to the functor to handle the exception.
       */
      template <typename O>
      [[nodiscard]] constexpr auto then(O&& iO) {
        return _then(std::forward<O>(iO), std::move(*this));
      }

      ///Define next task to run once this task has finished. If a previous task had an exception, only the first functor, iE, will be run and passed the std::exception_ptr const. If there
      /// was no exception, then only the functor iF will be run. If iF has an exception, it will be automatically propagated to the edm::WaitingTaskHolder.
      template <typename E, typename F>
      [[nodiscard]] constexpr auto thenIfExceptionElse(E&& iE, F&& iF) {
        return WaitingTaskChain<ExplicitExceptionHandler<E, F>, Conditional<U>, T...>(
            ExplicitExceptionHandler<E, F>(std::forward<E>(iE), std::forward<F>(iF)), std::move(*this));
      }

      template <typename O>
      [[nodiscard]] constexpr auto ifThen(bool iCondition, O&& iO) {
        return WaitingTaskChain<Conditional<AutoExceptionHandler<O>>, Conditional<U>, T...>(
            iCondition, AutoExceptionHandler<O>(std::forward<O>(iO)), std::move(*this));
      }

      [[nodiscard]] WaitingTaskHolder task(WaitingTaskHolder iV) {
        if (condition_) {
          return l_.task(WaitingTaskHolder(
              *iV.group(),
              make_waiting_task([f = std::move(f_), v = std::move(iV)](std::exception_ptr const* iPtr) mutable {
                f(iPtr, std::move(v));
              })));
        }
        return l_.task(iV);
      }

      void run(WaitingTaskHolder iV) {
        if (condition_) {
          l_.run(WaitingTaskHolder(
              *iV.group(),
              make_waiting_task([f = std::move(f_), v = std::move(iV)](std::exception_ptr const* iPtr) mutable {
                f(iPtr, std::move(v));
              })));
        } else {
          l_.run(iV);
        }
      }

    private:
      WaitingTaskChain<T...> l_;
      U f_;
      bool condition_;
    };

  }  // namespace waiting_task::detail
  namespace waiting_task {
    template <typename F>
    auto beginChain(F&& iF) {
      using namespace detail;
      if constexpr (has_exception_handling<F>::value) {
        return WaitingTaskChain<F>(std::forward<F>(iF));
      } else {
        return WaitingTaskChain<AutoExceptionHandler<F>>(AutoExceptionHandler<F>(std::forward<F>(iF)));
      }
    }

    template <typename E, typename F>
    auto beginChainIfExceptionElse(E&& iE, F&& iF) {
      using namespace detail;
      return WaitingTaskChain<ExplicitExceptionHandler<E, F>>(
          ExplicitExceptionHandler<E, F>(std::forward<E>(iE), std::forward<F>(iF)));
    }
  }  // namespace waiting_task
}  // namespace edm

#endif
