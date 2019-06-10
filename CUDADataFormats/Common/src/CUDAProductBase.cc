#include "CUDADataFormats/Common/interface/CUDAProductBase.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

bool CUDAProductBase::isAvailable() const {
  // In absence of event, the product was available already at the end
  // of produce() of the producer.
  if(not event_) {
    return true;
  }
  return event_->has_occurred();
}
