//
//  test_catch2_WaitingTaskChain.cpp
//  CMSSW
//
//  Created by Chris Jones on 7/8/21.
//

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "tbb/global_control.h"

#include "FWCore/Concurrency/interface/beginWaitingTaskChain.h"

TEST_CASE("Test beginWaitingTaskChain", "[beginWaitingTaskChain]") {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);

  SECTION("no explicit exception handling") {
    SECTION("begin.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::beginWaitingTaskChain([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 }).end(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::beginWaitingTaskChain([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .next([&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .end(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::beginWaitingTaskChain([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .next([&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .next([&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 3);
                     })
                     .end(edm::WaitingTaskHolder(group, &waitTask));

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
        edm::beginWaitingTaskChain([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 1);
        })
            .next([&count](auto h) {
              ++count;
              REQUIRE(count.load() == 2);
            })
            .next([&count](auto h) {
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
        auto h = edm::beginWaitingTaskChain([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(false);
                 }).end(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::beginWaitingTaskChain([&count](edm::WaitingTaskHolder h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                   throw std::exception();
                 }).end(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::beginWaitingTaskChain([&count](auto h) {
                   ++count;
                   REQUIRE(count.load() == 1);
                   throw std::exception();
                 })
                     .next([&count](auto h) {
                       ++count;
                       REQUIRE(false);
                     })
                     .next([&count](auto h) {
                       ++count;
                       REQUIRE(false);
                     })
                     .end(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::exception_ptr());
      }
      group.wait();
      REQUIRE(count.load() == 1);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() != nullptr);
    }
  }

  SECTION("nextWithException testing") {
    SECTION("begin.end") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h =
            edm::beginWaitingTaskChainWithException([&count](std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) {
              REQUIRE(iPtr == nullptr);
              ++count;
              REQUIRE(count.load() == 1);
            }).end(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::beginWaitingTaskChainWithException([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .nextWithException([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr == nullptr);
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .end(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::beginWaitingTaskChainWithException([&count](std::exception_ptr const* iPtr, auto h) {
                   REQUIRE(iPtr == nullptr);
                   ++count;
                   REQUIRE(count.load() == 1);
                 })
                     .nextWithException([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr == nullptr);
                       ++count;
                       REQUIRE(count.load() == 2);
                     })
                     .nextWithException([&count](std::exception_ptr const* iPtr, auto h) {
                       REQUIRE(iPtr == nullptr);
                       ++count;
                       REQUIRE(count.load() == 3);
                     })
                     .end(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h =
            edm::beginWaitingTaskChainWithException([&count](std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) {
              REQUIRE(iPtr != nullptr);
              ++count;
              REQUIRE(count.load() == 1);
            }).end(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h =
            edm::beginWaitingTaskChainWithException([&count](std::exception_ptr const* iPtr, edm::WaitingTaskHolder h) {
              REQUIRE(iPtr != nullptr);
              ++count;
              REQUIRE(count.load() == 1);
              h.doneWaiting(*iPtr);
            })
                .nextWithException([&count](std::exception_ptr const* iPtr, auto h) {
                  REQUIRE(iPtr != nullptr);
                  ++count;
                  REQUIRE(count.load() == 2);
                })
                .end(edm::WaitingTaskHolder(group, &waitTask));

        h.doneWaiting(std::make_exception_ptr(std::exception()));
      }
      group.wait();
      REQUIRE(count.load() == 2);
      REQUIRE(waitTask.done());
      REQUIRE(waitTask.exceptionPtr() == nullptr);
    }
  }

  SECTION("ifExceptionElseNext testing") {
    SECTION("begin.end") {
      std::atomic<int> count{0};
      std::atomic<int> exceptCount{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        auto h = edm::beginWaitingTaskChainIfExceptionElseNext(
                     [&exceptCount](std::exception_ptr const& iPtr) {
                       ++exceptCount;
                       REQUIRE(false);
                     },
                     [&count](edm::WaitingTaskHolder h) {
                       ++count;
                       REQUIRE(count.load() == 1);
                     })
                     .end(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::beginWaitingTaskChainIfExceptionElseNext(
                     [&exceptCount](std::exception_ptr const& iPtr) {
                       ++exceptCount;
                       REQUIRE(false);
                     },
                     [&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 1);
                     })
                     .ifExceptionElseNext(
                         [&exceptCount](std::exception_ptr const& iPtr) {
                           ++exceptCount;
                           REQUIRE(false);
                         },
                         [&count](auto h) {
                           ++count;
                           REQUIRE(count.load() == 2);
                         })
                     .end(edm::WaitingTaskHolder(group, &waitTask));

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
        auto h = edm::beginWaitingTaskChainIfExceptionElseNext(
                     [&exceptCount](std::exception_ptr const& iPtr) {
                       ++exceptCount;
                       REQUIRE(false);
                     },
                     [&count](auto h) {
                       ++count;
                       REQUIRE(count.load() == 1);
                     })
                     .ifExceptionElseNext(
                         [&exceptCount](std::exception_ptr const& iPtr) {
                           ++exceptCount;
                           REQUIRE(false);
                         },
                         [&count](auto h) {
                           ++count;
                           REQUIRE(count.load() == 2);
                         })
                     .ifExceptionElseNext(
                         [&exceptCount](std::exception_ptr const& iPtr) {
                           ++exceptCount;
                           REQUIRE(false);
                         },
                         [&count](auto h) {
                           ++count;
                           REQUIRE(count.load() == 3);
                         })
                     .end(edm::WaitingTaskHolder(group, &waitTask));

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
      auto h = edm::beginWaitingTaskChainIfExceptionElseNext(
                   [&exceptCount](std::exception_ptr const& iPtr) {
                     ++exceptCount;
                     REQUIRE(exceptCount.load() == 1);
                   },
                   [&count](auto h) {
                     ++count;
                     REQUIRE(false);
                   })
                   .ifExceptionElseNext(
                       [&exceptCount](std::exception_ptr const& iPtr) {
                         ++exceptCount;
                         REQUIRE(exceptCount.load() == 2);
                       },
                       [&count](auto h) {
                         ++count;
                         REQUIRE(false);
                       })
                   .ifExceptionElseNext(
                       [&exceptCount](std::exception_ptr const& iPtr) {
                         ++exceptCount;
                         REQUIRE(exceptCount.load() == 3);
                       },
                       [&count](auto h) {
                         ++count;
                         REQUIRE(false);
                       })
                   .end(edm::WaitingTaskHolder(group, &waitTask));

      h.doneWaiting(std::make_exception_ptr(std::exception()));
    }
    group.wait();
    REQUIRE(exceptCount.load() == 3);
    REQUIRE(count.load() == 0);
    REQUIRE(waitTask.done());
    REQUIRE(waitTask.exceptionPtr() != nullptr);
  }
  SECTION("ifThenNext testing") {
    SECTION("begin.ifTheNext(true).next.run") {
      std::atomic<int> count{0};

      edm::FinalWaitingTask waitTask;
      tbb::task_group group;
      {
        edm::beginWaitingTaskChain([&count](auto h) {
          ++count;
          REQUIRE(count.load() == 1);
        })
            .ifThenNext(true,
                        [&count](auto h) {
                          ++count;
                          REQUIRE(count.load() == 2);
                        })
            .next([&count](auto h) {
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

    SECTION("ifThenNext testing") {
      SECTION("begin.ifTheNext(false).next.run") {
        std::atomic<int> count{0};

        edm::FinalWaitingTask waitTask;
        tbb::task_group group;
        {
          edm::beginWaitingTaskChain([&count](auto h) {
            ++count;
            REQUIRE(count.load() == 1);
          })
              .ifThenNext(false,
                          [&count](auto h) {
                            ++count;
                            REQUIRE(false);
                          })
              .next([&count](auto h) {
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
