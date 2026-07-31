#ifndef FWCore_Utilities_SignalSentry_h
#define FWCore_Utilities_SignalSentry_h

#include <optional>
#include <type_traits>
#include <utility>

#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace edm::signalslot {
  /**
   * This is a sentry class intended to be used to emit Signal objects in a way that guarantees that the Signal is
   * emitted even if an exception is thrown.  The sentry is constructed with a callable object that will be called in
   * the destructor unless the succeeded() method is called first.  The callable object should be a lambda that emits
   * the Signal.
   *
   * The user is expected to call the succeeded() method to emit the Signal if the operation that is being guarded by
   * the sentry is successful. If the Signal throws an exception in this case, it is propagated from the succeeded()
   * method.
   *
   * If the guarded operation throws an exception, the Signal is emitted from the destructor of the sentry, and any
   * exceptions from the Signal are ignored. This is the behavior also if the user forgets to call the succeeded()
   * method.
   */
  template <typename F>
    requires std::is_invocable_v<F>
  class SignalSentry {
  public:
    SignalSentry(F iFunc) : func_(std::move(iFunc)) {}
    SignalSentry(SignalSentry const&) = delete;
    SignalSentry& operator=(SignalSentry const&) = delete;
    SignalSentry(SignalSentry&&) = delete;
    SignalSentry& operator=(SignalSentry&&) = delete;

    ~SignalSentry() {
      if (func_) {
        // Must assume an exception is already in flight, so must ignore any exceptions thrown by func_.
        CMS_SA_ALLOW try { (*func_)(); } catch (...) {
        }
      }
    }

    void succeeded() {
      (*func_)();
      func_.reset();
    }

  private:
    std::optional<F> func_;
  };

  template <typename F>
  auto make_sentry(F&& iFunc) {
    return SignalSentry<F>(std::forward<F>(iFunc));
  }
}  // namespace edm::signalslot

#endif
