#ifndef FWCore_Concurrency_chain_first_h
#define FWCore_Concurrency_chain_first_h
// -*- C++ -*-
//
// Package:     Concurrency
// function  :     edm::waiting_task::chain::first
//
/**\function chain_first

 Description: Handles creation of a chain of WaitingTasks

 Usage:
    Use the function to begin constructing a chain of waiting tasks.
    Once the chain is declared, call lastTask() with a WaitingTaskHolder
    to get back a new WaitingTaskHolder or runLast() to schedule the chain to run.
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
#include "FWCore/Utilities/interface/thread_safety_macros.h"

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
          CMS_SA_ALLOW try { f_(h); } catch (...) {
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
          CMS_SA_ALLOW try { f_(h); } catch (...) {
            h.doneWaiting(std::current_exception());
          }
        }
      }

    private:
      E except_;
      F f_;
    };

    /**Creates a functor adaptor which assembled two different functors into one. To use, one calls the constructor immediately followed by the else_ method. The created functor has the following behavior:
 If a previous task had an exception, only the first functor given to the constructor, iE, will be run and passed the std::exception_ptr const. If there
 was no exception, then only the functor passed to else_, iF, will be run. If iF has an exception, it will be automatically propagated to the edm::WaitingTaskHolder. */
    template <typename E>
    struct IfExceptionAdapter {
      constexpr IfExceptionAdapter(E&& iE) : except_(std::forward<E>(iE)) {}

      template <typename F>
      constexpr auto else_(F&& iF) {
        return ExplicitExceptionHandler<E, F>(std::move(except_), std::forward<F>(iF));
      }

    private:
      E except_;
    };

    template <typename... T>
    struct WaitingTaskChain;

    template <typename F>
    struct Conditional {};

    template <typename F>
    struct ConditionalAdaptor {
      constexpr explicit ConditionalAdaptor(bool iCond, F&& iF) : f_(std::forward<F>(iF)), condition_(iCond) {}

      template <typename... T>
      [[nodiscard]] constexpr auto pipe(WaitingTaskChain<T...> iChain) {
        return WaitingTaskChain<Conditional<F>, T...>(condition_, std::move(f_), std::move(iChain));
      }

      F f_;
      bool condition_;
    };

    template <typename F>
    struct ThenAdaptor {
      constexpr explicit ThenAdaptor(F&& iF) : f_(std::forward<F>(iF)) {}

      template <typename... T>
      [[nodiscard]] constexpr auto pipe(WaitingTaskChain<T...> iChain) {
        return WaitingTaskChain<F, T...>(std::move(f_), std::move(iChain));
      }

    private:
      F f_;
    };

    struct RunLastAdaptor {
      explicit RunLastAdaptor(edm::WaitingTaskHolder iT) : task_(std::move(iT)) {}

      template <typename... T>
      constexpr void pipe(WaitingTaskChain<T...>&& iChain) {
        iChain.runLast(std::move(task_));
      }

    private:
      edm::WaitingTaskHolder task_;
    };

    struct LastTaskAdaptor {
      explicit LastTaskAdaptor(edm::WaitingTaskHolder iT) : task_(std::move(iT)) {}

      template <typename... T>
      constexpr auto pipe(WaitingTaskChain<T...>&& iChain) {
        return iChain.lastTask(std::move(task_));
      }

    private:
      edm::WaitingTaskHolder task_;
    };

    template <typename U>
    struct WaitingTaskChain<U> {
      constexpr explicit WaitingTaskChain(U&& iU) : f_{std::forward<U>(iU)} {}

      [[nodiscard]] WaitingTaskHolder lastTask(WaitingTaskHolder iV) {
        return WaitingTaskHolder(
            *iV.group(),
            make_waiting_task([f = std::move(f_), v = std::move(iV)](const std::exception_ptr* iPtr) mutable {
              f(iPtr, std::move(v));
            }));
      }

      void runLast(WaitingTaskHolder iH) { f_(nullptr, std::move(iH)); }

      template <typename V>
      friend auto operator|(WaitingTaskChain<U> iChain, V&& iV) {
        return iV.pipe(std::move(iChain));
      }

    private:
      U f_;
    };

    template <typename U, typename... T>
    struct WaitingTaskChain<U, T...> {
      explicit constexpr WaitingTaskChain(U&& iU, WaitingTaskChain<T...> iL)
          : l_(std::move(iL)), f_{std::forward<U>(iU)} {}

      [[nodiscard]] WaitingTaskHolder lastTask(WaitingTaskHolder iV) {
        return l_.lastTask(WaitingTaskHolder(
            *iV.group(),
            make_waiting_task([f = std::move(f_), v = std::move(iV)](std::exception_ptr const* iPtr) mutable {
              f(iPtr, std::move(v));
            })));
      }

      void runLast(WaitingTaskHolder iV) {
        l_.runLast(WaitingTaskHolder(
            *iV.group(),
            make_waiting_task([f = std::move(f_), v = std::move(iV)](std::exception_ptr const* iPtr) mutable {
              f(iPtr, std::move(v));
            })));
      }

      template <typename V>
      friend auto operator|(WaitingTaskChain<U, T...> iChain, V&& iV) {
        return iV.pipe(std::move(iChain));
      }

    private:
      WaitingTaskChain<T...> l_;
      U f_;
    };

    template <typename U, typename... T>
    struct WaitingTaskChain<Conditional<U>, T...> {
      explicit constexpr WaitingTaskChain(bool iCondition, U&& iU, WaitingTaskChain<T...> iL)
          : l_(std::move(iL)), f_{std::forward<U>(iU)}, condition_(iCondition) {}

      explicit constexpr WaitingTaskChain(Conditional<U> iC, WaitingTaskChain<T...> iL)
          : l_(std::move(iL)), f_{std::move<U>(iC.f_)}, condition_(iC.condition_) {}

      [[nodiscard]] WaitingTaskHolder lastTask(WaitingTaskHolder iV) {
        if (condition_) {
          return l_.lastTask(WaitingTaskHolder(
              *iV.group(),
              make_waiting_task([f = std::move(f_), v = std::move(iV)](std::exception_ptr const* iPtr) mutable {
                f(iPtr, std::move(v));
              })));
        }
        return l_.lastTask(iV);
      }

      void runLast(WaitingTaskHolder iV) {
        if (condition_) {
          l_.runLast(WaitingTaskHolder(
              *iV.group(),
              make_waiting_task([f = std::move(f_), v = std::move(iV)](std::exception_ptr const* iPtr) mutable {
                f(iPtr, std::move(v));
              })));
        } else {
          l_.runLast(iV);
        }
      }

      template <typename V>
      friend auto operator|(WaitingTaskChain<Conditional<U>, T...> iChain, V&& iV) {
        return iV.pipe(std::move(iChain));
      }

    private:
      WaitingTaskChain<T...> l_;
      U f_;
      bool condition_;
    };

  }  // namespace waiting_task::detail
  namespace waiting_task::chain {

    /** Sets the first task to be run as part of the task chain. The functor is expected to take either
   a single argument of type edm::WaitingTaskHolder or two arguments of type std::exception_ptr const* and WaitingTaskHolder. In the latter case, the pointer is only non-null if a previous task in the chain threw an exception.
   */
    template <typename F>
    [[nodiscard]] constexpr auto first(F&& iF) {
      using namespace detail;
      if constexpr (has_exception_handling<F>::value) {
        return WaitingTaskChain<F>(std::forward<F>(iF));
      } else {
        return WaitingTaskChain<AutoExceptionHandler<F>>(AutoExceptionHandler<F>(std::forward<F>(iF)));
      }
    }

    /**Define next task to run once this task has finished. Two different functor types are allowed
   1. The functor  takes a edm::WaitingTaskHolder as argument. If an exception happened in a previous task, the functor will NOT be run.
   If an exception happens while running the functor, the exception will be propagated to the WaitingTaskHolder.
   2. The functor takes a std::exception_ptr const* and WaitingTaskHolder. If an exception happened in a previous task, the first
   argument will be non-nullptr. In that case, the exception will NOT be automatically propagated to the WaitingTaskHolder. In addition,
   if the functor itself throws an exception, it is up to the functor to handle the exception.
   */
    template <typename O>
    [[nodiscard]] constexpr auto then(O&& iO) {
      using namespace detail;
      if constexpr (has_exception_handling<O>::value) {
        return ThenAdaptor<O>(std::forward<O>(iO));
      } else {
        return ThenAdaptor<AutoExceptionHandler<O>>(AutoExceptionHandler<O>(std::forward<O>(iO)));
      }
    }

    ///Only runs this task if the condition (which is known at the call time) is true. If false, this task will be skipped and the following task will run.
    template <typename O>
    [[nodiscard]] constexpr auto ifThen(bool iValue, O&& iO) {
      using namespace detail;
      if constexpr (has_exception_handling<O>::value) {
        return ConditionalAdaptor<O>(iValue, std::forward<O>(iO));
      } else {
        return ConditionalAdaptor<AutoExceptionHandler<O>>(iValue, AutoExceptionHandler<O>(std::forward<O>(iO)));
      }
    }

    [[nodiscard]] inline auto runLast(edm::WaitingTaskHolder iTask) { return detail::RunLastAdaptor(std::move(iTask)); }

    [[nodiscard]] inline auto lastTask(edm::WaitingTaskHolder iTask) {
      return detail::LastTaskAdaptor(std::move(iTask));
    }

    /**Creates a functor adaptor which assembled two different functors into one. To use, one calls the constructor immediately followed by the else_ method. The created functor has the following behavior:
 If a previous task had an exception, only the first functor given to the constructor, iE, will be run and passed the std::exception_ptr const. If there
 was no exception, then only the functor passed to else_, iF, will be run. If iF has an exception, it will be automatically propagated to the edm::WaitingTaskHolder. */
    template <typename E>
    [[nodiscard]] constexpr auto ifException(E&& iE) {
      return detail::IfExceptionAdapter(std::forward<E>(iE));
    }

  }  // namespace waiting_task::chain
}  // namespace edm

#endif
