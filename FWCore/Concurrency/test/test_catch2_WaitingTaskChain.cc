//
//  test_catch2_WaitingTaskChain.cpp
//  CMSSW
//
//  Created by Chris Jones on 7/8/21.
//

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "oneapi/tbb/global_control.h"

#include "FWCore/Concurrency/interface/chain_first.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

TEST_CASE("Test chain::first", "[chain::first]") {
  oneapi::tbb::global_control control(oneapi::tbb::global_control::max_allowed_parallelism, 1);

  SECTION("no explicit exception handling") {
    SECTION("first | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("first | then | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 }) |
                 then([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 2);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 2);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("first | then | then | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 }) |
                 then([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 2);
                 }) |
                 then([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 3);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 3);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("first | then | then | runLast") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        first([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 1);
        }) | then([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 2);
        }) | then([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 3);
        }) | runLast(edm::WaitingTaskHolder(group, &waitTask));
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 3);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("exception -> first | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(false);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::make_exception_ptr(std::exception()));
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 0);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr());
    }

    SECTION("first(exception) | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                   throw std::exception();
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr());
    }

    SECTION("first(exception) | then | then | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                   throw std::exception();
                 }) |
                 then([&count](auto h) {
                   ++count;
                   REQUIRE(false);
                 }) |
                 then([&count](auto h) {
                   ++count;
                   REQUIRE(false);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr());
    }
  }

  SECTION("then with exception handler testing") {
    SECTION("first | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("first | then | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 }) |
                 then([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 2);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 2);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("first | then | then | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 }) |
                 then([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 2);
                 }) |
                 then([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 3);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 3);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("exception -> first | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) {
                   REQUIRE(iPtr != nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::make_exception_ptr(std::exception()));
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("exception -> first | then | lastTask") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first([&count](std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) {
                   REQUIRE(iPtr != nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                   h.doneWaiting(*iPtr);
                 }) |
                 then([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr != nullptr);
                   ++count;
                   REQUIRE(count.load() == 2);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::make_exception_ptr(std::exception()));
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 2);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }
  }

  SECTION("ifException.else testing") {
    SECTION("first | lastTask") {
      std::atomic<int> count{0};
      std::atomic<int> exceptCount{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                         ++exceptCount;
                         REQUIRE(false);
                       }).else_([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 })) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(exceptCount.load() == 0);
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("first | then | lastTask") {
      std::atomic<int> count{0};
      std::atomic<int> exceptCount{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                         ++exceptCount;
                         REQUIRE(false);
                       }).else_([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 })) |
                 then(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                        ++exceptCount;
                        REQUIRE(false);
                      }).else_([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 2);
                 })) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(exceptCount.load() == 0);
      REQUIRE(count.load() == 2);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("first | then | then | lastTask") {
      std::atomic<int> count{0};
      std::atomic<int> exceptCount{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                         ++exceptCount;
                         REQUIRE(false);
                       }).else_([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 })) |
                 then(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                        ++exceptCount;
                        REQUIRE(false);
                      }).else_([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 2);
                 })) |
                 then(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                        ++exceptCount;
                        REQUIRE(false);
                      }).else_([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 3);
                 })) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      waitTask.waitNoThrow();
      REQUIRE(exceptCount.load() == 0);
      REQUIRE(count.load() == 3);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("exception -> first | then | then | lastTask") {
      std::atomic<int> count{0};
      std::atomic<int> exceptCount{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        auto h = first(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                         ++exceptCount;
                         REQUIRE(exceptCount.load() == 1);
                       }).else_([&count](auto h) {
                   ++count;
                   REQUIRE(false);
                 })) |
                 then(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                        ++exceptCount;
                        REQUIRE(exceptCount.load() == 2);
                      }).else_([&count](auto h) {
                   ++count;
                   REQUIRE(false);
                 })) |
                 then(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                        ++exceptCount;
                        REQUIRE(exceptCount.load() == 3);
                      }).else_([&count](auto h) {
                   ++count;
                   REQUIRE(false);
                 })) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::make_exception_ptr(std::exception()));
      }
      waitTask.waitNoThrow();
      REQUIRE(exceptCount.load() == 3);
      REQUIRE(count.load() == 0);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr());
    }
  }

  SECTION("ifThen testing") {
    SECTION("first | ifThen(true) | then | runLast") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        first([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 1);
        }) | ifThen(true, [&count](auto h) {
          ++count;
          REQUIRE(count.load() == 2);
        }) | then([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 3);
        }) | runLast(edm::WaitingTaskHolder(group, &waitTask));
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 3);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }

    SECTION("first | ifThen(false) | then | runLast") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};
      {
        using namespace edm::waiting_task::chain;
        first([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 1);
        }) | ifThen(false, [&count](auto h) {
          ++count;
          REQUIRE(false);
        }) | then([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 2);
        }) | runLast(edm::WaitingTaskHolder(group, &waitTask));
      }
      waitTask.waitNoThrow();
      REQUIRE(count.load() == 2);
      REQUIRE(waitTask.done());
      REQUIRE(not waitTask.exceptionPtr());
    }
  }
}
