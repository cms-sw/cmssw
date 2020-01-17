#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace cms::cuda {
  void ContextState::throwIfStream() const {
    if (stream_) {
      throw cms::Exception("LogicError") << "Trying to set ContextState, but it already had a valid state";
    }
  }

  void ContextState::throwIfNoStream() const {
    if (not stream_) {
      throw cms::Exception("LogicError") << "Trying to get ContextState, but it did not have a valid state";
    }
  }
}  // namespace cms::cuda
