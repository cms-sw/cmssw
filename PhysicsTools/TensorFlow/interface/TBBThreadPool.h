/*
 * Custom TensorFlow thread pool implementation that schedules tasks in TBB.
 * Based on TensorFlow 2.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_TBBTHREADPOOL_H
#define PHYSICSTOOLS_TENSORFLOW_TBBTHREADPOOL_H

#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "tensorflow/core/lib/core/threadpool.h"

#include "tbb/task_arena.h"
#include "tbb/task_group.h"
#include "tbb/global_control.h"

namespace tensorflow {

  class TBBThreadPool : public tensorflow::thread::ThreadPoolInterface {
  public:
    static TBBThreadPool& instance(int nThreads = -1) {
      CMS_THREAD_SAFE static TBBThreadPool pool(nThreads);
      return pool;
    }

    explicit TBBThreadPool(int nThreads = -1)
        : nThreads_(nThreads > 0 ? nThreads
                                 : tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)),
          numScheduleCalled_(0) {
      // when nThreads is zero or smaller, use the default value determined by tbb
    }

    void Schedule(std::function<void()> fn) override {
      numScheduleCalled_ += 1;

      // use a task arena to avoid having unrelated tasks start
      // running on this thread, which could potentially start deadlocks
      tbb::task_arena taskArena;
      tbb::task_group taskGroup;

      // we are required to always call wait before destructor
      auto doneWithTaskGroup = [&taskArena, &taskGroup](void*) {
        taskArena.execute([&taskGroup]() { taskGroup.wait(); });
      };
      std::unique_ptr<tbb::task_group, decltype(doneWithTaskGroup)> taskGuard(&taskGroup, doneWithTaskGroup);

      // schedule the task
      taskArena.execute([&taskGroup, &fn] { taskGroup.run(fn); });

      // reset the task guard which will call wait
      taskGuard.reset();
    }

    void ScheduleWithHint(std::function<void()> fn, int start, int end) override { Schedule(fn); }

    void Cancel() override {}

    int NumThreads() const override { return nThreads_; }

    int CurrentThreadId() const override {
      static std::atomic<int> idCounter{0};
      thread_local const int id = idCounter++;
      return id;
    }

    int GetNumScheduleCalled() { return numScheduleCalled_; }

  private:
    const int nThreads_;
    std::atomic<int> numScheduleCalled_;
  };

}  // namespace tensorflow

#endif  // PHYSICSTOOLS_TENSORFLOW_TBBTHREADPOOL_H
