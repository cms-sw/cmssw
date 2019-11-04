#include "CUDADataFormats/Common/interface/CUDAProductBase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/eventIsOccurred.h"

bool CUDAProductBase::isAvailable() const {
  // In absence of event, the product was available already at the end
  // of produce() of the producer.
  if (not event_) {
    return true;
  }
  return cudautils::eventIsOccurred(event_->id());
}

CUDAProductBase::~CUDAProductBase() {
  // Make sure that the production of the product in the GPU is
  // complete before destructing the product. This is to make sure
  // that the EDM stream does not move to the next event before all
  // asynchronous processing of the current is complete.
  if (event_) {
    // TODO: a callback notifying a WaitingTaskHolder (or similar)
    // would avoid blocking the CPU, but would also require more work.
    event_->synchronize();
  }
}
