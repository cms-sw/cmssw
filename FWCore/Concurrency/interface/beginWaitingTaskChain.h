#ifndef FWCore_Concurrency_beginWaitingTaskChain_h
#define FWCore_Concurrency_beginWaitingTaskChain_h
// -*- C++ -*-
//
// Package:     Concurrency
// function  :     beginWaitingTaskChain
//
/**\function beginWaitingTaskChain

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

// user include files
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

// forward declarations

namespace edm {
  namespace task_chain::detail {

    template <typename... T>
    struct WaitingTaskChain;

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

    template <typename U>
    struct WaitingTaskChain<U> {
      constexpr explicit WaitingTaskChain(U&& iU) : f_{std::forward<U>(iU)} {}

      ///Define next task to run once this task has finished. The functor must take a edm::WaitingTaskHolder as argument. If an exception happened
      /// in a previous task, the functor will NOT be run. If an exception happens while running the functor, the exception will be propagated to the
      /// WaitingTaskHolder.
      template <typename O>
      [[nodiscard]] constexpr WaitingTaskChain<AutoExceptionHandler<O>, U> next(O&& iO) {
        return WaitingTaskChain<AutoExceptionHandler<O>, U>(AutoExceptionHandler<O>(std::forward<O>(iO)),
                                                            std::move(*this));
      }

      ///Define next task to run once this task has finished. The functor must take std::exception_ptr const* and a  edm::WaitingTaskHolder as arguments
      template <typename O>
      [[nodiscard]] constexpr auto nextWithException(O&& iO) {
        return WaitingTaskChain<O, U>(std::forward<O>(iO), std::move(*this));
      }

      ///Define next task to run once this task has finished. If a previous task had an exception, only the first functor, iE, will be run and passed the std::exception_ptr const. If there
      /// was no exception, then only the functor iF will be run. If iF has an exception, it will be automatically propagated to the edm::WaitingTaskHolder.
      template <typename E, typename F>
      [[nodiscard]] constexpr auto ifExceptionElseNext(E&& iE, F&& iF) {
        return WaitingTaskChain<ExplicitExceptionHandler<E, F>, U>(
            ExplicitExceptionHandler<E, F>(std::forward<E>(iE), std::forward<F>(iF)), std::move(*this));
      }

      [[nodiscard]] WaitingTaskHolder end(WaitingTaskHolder iV) {
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

      ///Define next task to run once this task has finished. The functor must take a edm::WaitingTaskHolder as argument. If an exception happened
      /// in a previous task, the functor will NOT be run.
      template <typename O>
      [[nodiscard]] constexpr auto next(O&& iO) {
        return WaitingTaskChain<AutoExceptionHandler<O>, U, T...>(AutoExceptionHandler<O>(std::forward<O>(iO)),
                                                                  std::move(*this));
      }

      ///Define next task to run once this task has finished. The functor must take std::exception_ptr const* and a  edm::WaitingTaskHolder as arguments
      template <typename O>
      [[nodiscard]] constexpr auto nextWithException(O&& iO) {
        return WaitingTaskChain<O, U, T...>(std::forward<O>(iO), std::move(*this));
      }

      ///Define next task to run once this task has finished. If a previous task had an exception, only the first functor, iE, will be run and passed the std::exception_ptr const. If there
      /// was no exception, then only the functor iF will be run. If iF has an exception, it will be automatically propagated to the edm::WaitingTaskHolder.
      template <typename E, typename F>
      [[nodiscard]] constexpr auto ifExceptionElseNext(E&& iE, F&& iF) {
        return WaitingTaskChain<ExplicitExceptionHandler<E, F>, U, T...>(
            ExplicitExceptionHandler<E, F>(std::forward<E>(iE), std::forward<F>(iF)), std::move(*this));
      }

      [[nodiscard]] WaitingTaskHolder end(WaitingTaskHolder iV) {
        return l_.end(WaitingTaskHolder(
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
  }  // namespace task_chain::detail
  template <typename F>
  auto beginWaitingTaskChain(F&& iF) {
    using namespace task_chain::detail;
    return WaitingTaskChain<AutoExceptionHandler<F>>(AutoExceptionHandler<F>(std::forward<F>(iF)));
  }

  template <typename F>
  auto beginWaitingTaskChainWithException(F&& iF) {
    using namespace task_chain::detail;
    return WaitingTaskChain<F>(std::forward<F>(iF));
  }
  template <typename E, typename F>
  auto beginWaitingTaskChainIfExceptionElseNext(E&& iE, F&& iF) {
    using namespace task_chain::detail;
    return WaitingTaskChain<ExplicitExceptionHandler<E, F>>(
        ExplicitExceptionHandler<E, F>(std::forward<E>(iE), std::forward<F>(iF)));
  }

}  // namespace edm

#endif
