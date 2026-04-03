#include <memory>

#include "DataFormats/AlpakaCommon/interface/alpaka/EDMetadata.h"
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataAcquireSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/EventCache.h"
#include "HeterogeneousCore/AlpakaInterface/interface/QueueCache.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace detail {
    EDMetadataAcquireSentry::EDMetadataAcquireSentry(edm::StreamID streamID,
                                                     edm::WaitingTaskWithArenaHolder holder,
                                                     bool synchronize)
        : EDMetadataAcquireSentry(cms::alpakatools::getQueueCache<Queue>().get(detail::chooseDevice(streamID)),
                                  std::move(holder),
                                  synchronize) {}

    EDMetadataAcquireSentry::EDMetadataAcquireSentry(std::shared_ptr<Queue> queue,
                                                     edm::WaitingTaskWithArenaHolder holder,
                                                     bool synchronize)
        : waitingTaskHolder_(std::move(holder)), synchronize_(synchronize) {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
      // all synchronous backends
      metadata_ = std::make_shared<EDMetadata>(std::move(queue));
#else
      // all asynchronous backends
      const Device& device = alpaka::getDev(*queue);
      metadata_ = std::make_shared<EDMetadata>(std::move(queue), cms::alpakatools::getEventCache<Event>().get(device));
#endif
    }

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
