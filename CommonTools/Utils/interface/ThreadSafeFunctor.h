#ifndef CommonTools_Utils_ThreadSafeFunctor_H
#define CommonTools_Utils_ThreadSafeFunctor_H

#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <mutex>
#include <utility>

// This class is a simple wrapper around some functor class to use its operator() in a thread-safe way.

template <class Functor>
class ThreadSafeFunctor {
public:
  template <typename... Params>
  ThreadSafeFunctor(Params&&... params) : functor_{std::forward<Params>(params)...} {}

  ThreadSafeFunctor(ThreadSafeFunctor&& other) noexcept : functor_(std::move(other.functor_)) {}

  template <typename... Params>
  typename std::invoke_result_t<Functor, Params...> operator()(Params&&... params) const {
    std::lock_guard<std::mutex> guard(mutex_);
    return functor_(std::forward<Params>(params)...);
  }

private:
  const Functor functor_;
  CMS_THREAD_SAFE mutable std::mutex mutex_;
};

#endif
