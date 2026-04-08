#include <memory>

#include "DataFormats/AlpakaCommon/interface/alpaka/EDMetadata.h"
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataAcquireSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace detail {
    EDMetadataAcquireSentry::EDMetadataAcquireSentry(edm::StreamID streamID,
                                                     edm::WaitingTaskWithArenaHolder holder,
                                                     bool synchronize)
        : metadata_(std::make_shared<EDMetadata>(detail::chooseDevice(streamID))),
          waitingTaskHolder_(std::move(holder)),
          synchronize_(synchronize) {}

    EDMetadataAcquireSentry::EDMetadataAcquireSentry(std::shared_ptr<Queue> queue,
                                                     edm::WaitingTaskWithArenaHolder holder,
                                                     bool synchronize)
        : metadata_(std::make_shared<EDMetadata>(std::move(queue))),
          waitingTaskHolder_(std::move(holder)),
          synchronize_(synchronize) {}

#ifndef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    // all asynchronous backends
    std::shared_ptr<EDMetadata> EDMetadataAcquireSentry::finish() {
      if (synchronize_) {
        alpaka::wait(metadata_->queue());
      } else {
        asyncWait();
      }
      return std::move(metadata_);
    }

    // all asynchronous backends
    void EDMetadataAcquireSentry::asyncWait() {
      edm::Service<edm::Async> async;
      auto event = metadata_->recordEvent();
      // wait for the event to be ready in an async thread, then notify the waitingTaskHolder_
      async->runAsync(
          std::move(waitingTaskHolder_),
          [event = std::move(event)]() mutable { alpaka::wait(*event); },
          []() {
            return "Enqueued via " EDM_STRINGIZE(
                ALPAKA_ACCELERATOR_NAMESPACE) "::detail::EDMetadataAcquireSentry::asyncWait()";
          });
    }
#endif

  }  // namespace detail
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
