/*
 * Custom TensorFlow thread pool implementation that does no threading at all,
 * but schedules all tasks in the caller thread.
 * Based on TensorFlow 2.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_NOTHREADPOOL_H
#define PHYSICSTOOLS_TENSORFLOW_NOTHREADPOOL_H

#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/threadpool_options.h"

namespace tensorflow {

  class NoThreadPool : public tensorflow::thread::ThreadPoolInterface {
  public:
    static NoThreadPool& instance() {
      CMS_THREAD_SAFE static NoThreadPool pool;
      return pool;
    }

    explicit NoThreadPool() : numScheduleCalled_(0) {}

    void Schedule(std::function<void()> fn) override {
      numScheduleCalled_ += 1;
      fn();
    }

    void ScheduleWithHint(std::function<void()> fn, int start, int end) override { Schedule(fn); }

    void Cancel() override {}

    int NumThreads() const override { return 1; }

    int CurrentThreadId() const override { return -1; }

    int GetNumScheduleCalled() { return numScheduleCalled_; }

  private:
    std::atomic<int> numScheduleCalled_;
  };

}  // namespace tensorflow

#endif  // PHYSICSTOOLS_TENSORFLOW_NOTHREADPOOL_H
