#ifndef RecoEgamma_EgammaTools_ThreadSafeStringCut_H
#define RecoEgamma_EgammaTools_ThreadSafeStringCut_H

#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <mutex>
#include "FWCore/Utilities/interface/thread_safety_macros.h"

/*
 * This class is a simple wrapper around either a StringObjectFunction or
 * StringCutObjectSelector to use them in a thread safe way.
 *
 */

template<class F, class T>
class ThreadSafeStringCut
{
  public:

    ThreadSafeStringCut(const std::string & expr) // constructor
      : func_(expr)
      , expr_(expr)
    {}

    ThreadSafeStringCut(ThreadSafeStringCut&& other) noexcept // move constructor
      : func_(std::move(other.func_))
      , expr_(std::move(other.expr_))
    {}

    typename std::result_of<F&(T)>::type operator()(const T & t) const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return func_(t);
    }

  private:

    const F func_;
    const std::string expr_;
    CMS_THREAD_SAFE mutable std::mutex mutex_;
};

#endif
