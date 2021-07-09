//
//  test_catch2_WaitingTaskChain.cpp
//  CMSSW
//
//  Created by Chris Jones on 7/8/21.
//

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "tbb/global_control.h"

#include "FWCore/Concurrency/interface/beginChain.h"

TEST_CASE("Test beginChain", "[beginChain]") {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);

  SECTION("no explicit exception handling") {
    SECTION("begin.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::beginChain([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 }).task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChain([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .then([&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChain([&count](auto h) {
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
                     .task(edm::WaitingTaskHolder(group, &waitTask));

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
        edm::waiting_task::beginChain([&count](auto h) {
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
            .run(edm::WaitingTaskHolder(group, &waitTask));
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
        auto h = edm::waiting_task::beginChain([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(false);
                 }).task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChain([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                   throw std::exception();
                 }).task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChain([&count](auto h) {
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
                     .task(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() != nullptr);
    }
  }

  SECTION("thenWithException testing") {
    SECTION("begin.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::waiting_task::beginChainWithException([&count](std::exception_ptr const* iPtr,
                                                                     edm::WaitingTaskHolder h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 }).task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChainWithException([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .thenWithException([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr == nullptr);
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChainWithException([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .thenWithException([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr == nullptr);
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .thenWithException([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr == nullptr);
                       ++count;
                       REQUIRE(count.load() == 3);
                     })
                     .task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChainWithException([&count](std::exception_ptr const* iPtr,
                                                                     edm::WaitingTaskHolder h) {
                   REQUIRE(iPtr != nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 }).task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChainWithException(
                     [&count](std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) {
                       REQUIRE(iPtr != nullptr);
                       ++count;
                       REQUIRE(count.load() == 1);
                       h.doneWaiting(*iPtr);
                     })
                     .thenWithException([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr != nullptr);
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChainIfExceptionElse(
                     [&exceptCount](std::exception_ptr const& iPtr) {
                       ++exceptCount;
                       REQUIRE(false);
                     },
                     [&count](edm::WaitingTaskHolder h) {
                       ++count;
                       REQUIRE(count.load() == 1);
                     })
                     .task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChainIfExceptionElse(
                     [&exceptCount](std::exception_ptr const& iPtr) {
                       ++exceptCount;
                       REQUIRE(false);
                     },
                     [&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 1);
                     })
                     .thenIfExceptionElse(
                         [&exceptCount](std::exception_ptr const& iPtr) {
                           ++exceptCount;
                           REQUIRE(false);
                         },
                         [&count](auto h) {
                           ++count;
                           REQUIRE(count.load() == 2);
                         })
                     .task(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::waiting_task::beginChainIfExceptionElse(
                     [&exceptCount](std::exception_ptr const& iPtr) {
                       ++exceptCount;
                       REQUIRE(false);
                     },
                     [&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 1);
                     })
                     .thenIfExceptionElse(
                         [&exceptCount](std::exception_ptr const& iPtr) {
                           ++exceptCount;
                           REQUIRE(false);
                         },
                         [&count](auto h) {
                           ++count;
                           REQUIRE(count.load() == 2);
                         })
                     .thenIfExceptionElse(
                         [&exceptCount](std::exception_ptr const& iPtr) {
                           ++exceptCount;
                           REQUIRE(false);
                         },
                         [&count](auto h) {
                           ++count;
                           REQUIRE(count.load() == 3);
                         })
                     .task(edm::WaitingTaskHolder(group, &waitTask));

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
      auto h = edm::waiting_task::beginChainIfExceptionElse(
                   [&exceptCount](std::exception_ptr const& iPtr) {
                     ++exceptCount;
                     REQUIRE(exceptCount.load() == 1);
                   },
                   [&count](auto h) {
                     ++count;
                     REQUIRE(false);
                   })
                   .thenIfExceptionElse(
                       [&exceptCount](std::exception_ptr const& iPtr) {
                         ++exceptCount;
                         REQUIRE(exceptCount.load() == 2);
                       },
                       [&count](auto h) {
                         ++count;
                         REQUIRE(false);
                       })
                   .thenIfExceptionElse(
                       [&exceptCount](std::exception_ptr const& iPtr) {
                         ++exceptCount;
                         REQUIRE(exceptCount.load() == 3);
                       },
                       [&count](auto h) {
                         ++count;
                         REQUIRE(false);
                       })
                   .task(edm::WaitingTaskHolder(group, &waitTask));

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
        edm::waiting_task::beginChain([&count](auto h) {
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
            .run(edm::WaitingTaskHolder(group, &waitTask));
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
          edm::waiting_task::beginChain([&count](auto h) {
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
              .run(edm::WaitingTaskHolder(group, &waitTask));
        }
        group.wait();
        REQUIRE(count.load() == 2);
        REQUIRE(waitTask.done());
        REQUIRE(waitTask.exceptionPtr() == nullptr);
      }
    }
  }
}
