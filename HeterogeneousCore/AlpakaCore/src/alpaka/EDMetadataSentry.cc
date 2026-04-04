#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/QueueCache.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace detail {

    EDMetadataSentry::EDMetadataSentry(std::shared_ptr<Queue> queue, bool synchronize) : synchronize_(synchronize) {
      assert(queue);
      metadata_ = std::make_shared<EDMetadata>(std::move(queue));
    }

    EDMetadataSentry::EDMetadataSentry(edm::StreamID streamID, bool synchronize) : synchronize_(synchronize) {
      auto const& device = detail::chooseDevice(streamID);
      metadata_ = std::make_shared<EDMetadata>(cms::alpakatools::getQueueCache<Queue>().get(device));
    }

    void EDMetadataSentry::finish(bool launchedAsyncWork) {
      if constexpr (not std::is_same_v<Queue, alpaka::Queue<Device, alpaka::Blocking>>) {
        if (launchedAsyncWork and synchronize_) {
          alpaka::wait(metadata_->queue());
        }
      }

      if (launchedAsyncWork and not synchronize_) {
        metadata_->recordEvent();
      }
    }

  }  // namespace detail
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
