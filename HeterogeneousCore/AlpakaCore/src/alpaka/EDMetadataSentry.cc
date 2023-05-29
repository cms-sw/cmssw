#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/EventCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/QueueCache.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace detail {
    EDMetadataSentry::EDMetadataSentry(edm::StreamID streamID) {
      auto const& device = detail::chooseDevice(streamID);
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
      metadata_ = std::make_shared<EDMetadata>(cms::alpakatools::getQueueCache<Queue>().get(device));
#else
      metadata_ = std::make_shared<EDMetadata>(cms::alpakatools::getQueueCache<Queue>().get(device),
                                               cms::alpakatools::getEventCache<Event>().get(device));
#endif
    }

    void EDMetadataSentry::finish() { metadata_->recordEvent(); }
  }  // namespace detail
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
