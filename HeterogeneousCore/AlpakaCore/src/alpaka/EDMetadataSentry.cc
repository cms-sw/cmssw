#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/EventCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/QueueCache.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace detail {
    EDMetadataSentry::EDMetadataSentry(edm::StreamID streamID, bool synchronize) : synchronize_(synchronize) {
      auto const& device = detail::chooseDevice(streamID);
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
      metadata_ = std::make_shared<EDMetadata>(cms::alpakatools::getQueueCache<Queue>().get(device));
#else
      metadata_ = std::make_shared<EDMetadata>(cms::alpakatools::getQueueCache<Queue>().get(device),
                                               cms::alpakatools::getEventCache<Event>().get(device));
#endif
    }

    void EDMetadataSentry::finish(bool launchedAsyncWork) {
      if constexpr (not std::is_same_v<Queue, alpaka::Queue<Device, alpaka::Blocking>>) {
        if (launchedAsyncWork and synchronize_) {
          alpaka::wait(metadata_->queue());
        }
      }

      if (launchedAsyncWork and not synchronize_) {
        metadata_->recordEvent();
      } else {
        // If we are certain no asynchronous work was launched (i.e.
        // the Queue was not used in any way), or a blocking
        // synchronization was explicitly requested, there is no need
        // to synchronize later, and the Event can be discarded.
        metadata_->discardEvent();
      }
    }
  }  // namespace detail
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
