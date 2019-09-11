#include "CUDADataFormats/Common/interface/CUDAProductBase.h"

bool CUDAProductBase::isAvailable() const {
  // In absence of event, the product was available already at the end
  // of produce() of the producer.
  if (not event_) {
    return true;
  }
  return event_->has_occurred();
}
