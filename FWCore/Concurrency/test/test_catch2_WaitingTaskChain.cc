//
//  test_catch2_WaitingTaskChain.cpp
//  CMSSW
//
//  Created by Chris Jones on 7/8/21.
//

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "tbb/global_control.h"

#include "FWCore/Concurrency/interface/chain_first.h"

TEST_CASE("Test chain::first", "[chain::first]") {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);

  SECTION("no explicit exception handling") {
    SECTION("begin.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 }).lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("begin.next.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .then([&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(count.load() == 2);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("begin.next.next.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .then([&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .then([&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 3);
                     })
                     .lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(count.load() == 3);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("begin.next.next.run") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        edm::waiting_task::chain::first([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 1);
        })
            .then([&count](auto h) {
              ++count;
              REQUIRE(count.load() == 2);
            })
            .then([&count](auto h) {
              ++count;
              REQUIRE(count.load() == 3);
            })
            .runLast(edm::WaitingTaskHolder(group, &waitTask));
      }
      group.wait();
      REQUIRE(count.load() == 3);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("exception -> begin.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(false);
                 }).lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::make_exception_ptr(std::exception()));
      }
      group.wait();
      REQUIRE(count.load() == 0);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() != nullptr);
    }

    SECTION("begin(exception).end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                   throw std::exception();
                 }).lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() != nullptr);
    }

    SECTION("begin(exception).next.next.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                   throw std::exception();
                 })
                     .then([&count](auto h) {
                       ++count;
                       REQUIRE(false);
                     })
                     .then([&count](auto h) {
                       ++count;
                       REQUIRE(false);
                     })
                     .lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() != nullptr);
    }
  }

  SECTION("then with exception handler testing") {
    SECTION("begin.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 }).lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("begin.next.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .then([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr == nullptr);
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(count.load() == 2);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("begin.next.next.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .then([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr == nullptr);
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .then([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr == nullptr);
                       ++count;
                       REQUIRE(count.load() == 3);
                     })
                     .lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(count.load() == 3);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("exception -> begin.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) {
                   REQUIRE(iPtr != nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 }).lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::make_exception_ptr(std::exception()));
      }
      group.wait();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("exception -> begin.next.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::chain::first([&count](std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) {
                   REQUIRE(iPtr != nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                   h.doneWaiting(*iPtr);
                 })
                     .then([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr != nullptr);
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::make_exception_ptr(std::exception()));
      }
      group.wait();
      REQUIRE(count.load() == 2);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }
  }

  SECTION("thenIfExceptionElse testing") {
    SECTION("begin.end") {
      std::atomic<int> count{0};
      std::atomic<int> exceptCount{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        using edm::waiting_task::chain::ifException;
        auto h = edm::waiting_task::chain::first(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                                                   ++exceptCount;
                                                   REQUIRE(false);
                                                 }).else_([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 })).lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(exceptCount.load() == 0);
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("begin.next.end") {
      std::atomic<int> count{0};
      std::atomic<int> exceptCount{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        using edm::waiting_task::chain::ifException;
        auto h = edm::waiting_task::chain::first(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                                                   ++exceptCount;
                                                   REQUIRE(false);
                                                 }).else_([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 }))
                     .then(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                             ++exceptCount;
                             REQUIRE(false);
                           }).else_([&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 2);
                     }))
                     .lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(exceptCount.load() == 0);
      REQUIRE(count.load() == 2);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("begin.next.next.end") {
      std::atomic<int> count{0};
      std::atomic<int> exceptCount{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        using edm::waiting_task::chain::ifException;
        auto h = edm::waiting_task::chain::first(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                                                   ++exceptCount;
                                                   REQUIRE(false);
                                                 }).else_([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 }))
                     .then(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                             ++exceptCount;
                             REQUIRE(false);
                           }).else_([&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 2);
                     }))
                     .then(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                             ++exceptCount;
                             REQUIRE(false);
                           }).else_([&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 3);
                     }))
                     .lastTask(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(exceptCount.load() == 0);
      REQUIRE(count.load() == 3);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }
  }

  SECTION("exception -> begin.next.next.end") {
    std::atomic<int> count{0};
    std::atomic<int> exceptCount{0};

    edm::FinalWaitingTask waitTask;
    tbb::task_group group;
    {
      using edm::waiting_task::chain::ifException;
      auto h = edm::waiting_task::chain::first(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                                                 ++exceptCount;
                                                 REQUIRE(exceptCount.load() == 1);
                                               }).else_([&count](auto h) {
                 ++count;
                 REQUIRE(false);
               }))
                   .then(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                           ++exceptCount;
                           REQUIRE(exceptCount.load() == 2);
                         }).else_([&count](auto h) {
                     ++count;
                     REQUIRE(false);
                   }))
                   .then(ifException([&exceptCount](std::exception_ptr const& iPtr) {
                           ++exceptCount;
                           REQUIRE(exceptCount.load() == 3);
                         }).else_([&count](auto h) {
                     ++count;
                     REQUIRE(false);
                   }))
                   .lastTask(edm::WaitingTaskHolder(group, &waitTask));

      h.doneWaiting(std::make_exception_ptr(std::exception()));
    }
    group.wait();
    REQUIRE(exceptCount.load() == 3);
    REQUIRE(count.load() == 0);
    REQUIRE(waitTask.done());
    REQUIRE(waitTask.exceptionPtr() != nullptr);
  }
  SECTION("ifThen testing") {
    SECTION("begin.ifTheNext(true).next.run") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        edm::waiting_task::chain::first([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 1);
        })
            .ifThen(true,
                    [&count](auto h) {
                      ++count;
                      REQUIRE(count.load() == 2);
                    })
            .then([&count](auto h) {
              ++count;
              REQUIRE(count.load() == 3);
            })
            .runLast(edm::WaitingTaskHolder(group, &waitTask));
      }
      group.wait();
      REQUIRE(count.load() == 3);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }

    SECTION("ifThen testing") {
      SECTION("begin.ifTheNext(false).next.run") {
        std::atomic<int> count{0};

        edm::FinalWaitingTask waitTask;
        tbb::task_group group;
        {
          edm::waiting_task::chain::first([&count](auto h) {
            ++count;
            REQUIRE(count.load() == 1);
          })
              .ifThen(false,
                      [&count](auto h) {
                        ++count;
                        REQUIRE(false);
                      })
              .then([&count](auto h) {
                ++count;
                REQUIRE(count.load() == 2);
              })
              .runLast(edm::WaitingTaskHolder(group, &waitTask));
        }
        group.wait();
        REQUIRE(count.load() == 2);
        REQUIRE(waitTask.done());
        REQUIRE(waitTask.exceptionPtr() == nullptr);
      }
    }
  }
}
