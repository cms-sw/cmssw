#ifndef FWCore_Concurrency_FinalWaitingTask_h
#define FWCore_Concurrency_FinalWaitingTask_h
// -*- C++ -*-
//
// Package:     FWCore/Concurrency
// Class  :     FinalWaitingTask
//
/**\class FinalWaitingTask FinalWaitingTask.h "FWCore/Concurrency/interface/FinalWaitingTask.h"

 Description: [one line class summary]

 Usage:
   Use this class on the stack to signal the final task to be run.
   Call done() to check to see if the task was run and check value of
   exceptionPtr() to see if an exception was thrown by any task in the group.

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 12 Jul 2022 18:45:15 GMT
//

// system include files
#include "oneapi/tbb/task_group.h"

// user include files
#include "FWCore/Concurrency/interface/WaitingTask.h"

// forward declarations
namespace edm {
  class FinalWaitingTask : public WaitingTask {
  public:
    FinalWaitingTask() = delete;
    explicit FinalWaitingTask(tbb::task_group& iGroup)
        : m_group{&iGroup}, m_handle{iGroup.defer([]() {})}, m_done{false} {}

    void execute() final { m_done = true; }

    [[nodiscard]] bool done() const noexcept { return m_done.load(); }

    void wait() {
      m_group->wait();
      if (exceptionPtr()) {
        std::rethrow_exception(exceptionPtr());
      }
    }
    std::exception_ptr waitNoThrow() {
      m_group->wait();
      return exceptionPtr();
    }

  private:
    void recycle() final { m_group->run(std::move(m_handle)); }
    tbb::task_group* m_group;
    tbb::task_handle m_handle;
    std::atomic<bool> m_done;
  };

}  // namespace edm
#endif
