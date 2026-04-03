#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::detail {

  EDMetadataSentry::EDMetadataSentry(edm::StreamID streamID, bool synchronize)
      : metadata_(std::make_shared<EDMetadata>(detail::chooseDevice(streamID))), synchronize_(synchronize) {}

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

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::detail
