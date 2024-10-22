#include "CUDADataFormats/Common/interface/ProductBase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/eventWorkHasCompleted.h"

namespace cms::cuda {
  bool ProductBase::isAvailable() const {
    // if default-constructed, the product is not available
    if (not event_) {
      return false;
    }
    return eventWorkHasCompleted(event_.get());
  }

  ProductBase::~ProductBase() {
    // Make sure that the production of the product in the GPU is
    // complete before destructing the product. This is to make sure
    // that the EDM stream does not move to the next event before all
    // asynchronous processing of the current is complete.

    // TODO: a callback notifying a WaitingTaskHolder (or similar)
    // would avoid blocking the CPU, but would also require more work.
    //
    // Intentionally not checking the return value to avoid throwing
    // exceptions. If this call would fail, we should get failures
    // elsewhere as well.
    if (event_) {
      cudaEventSynchronize(event_.get());
    }
  }
}  // namespace cms::cuda
