#include "CUDADataFormats/Common/interface/CUDAProductBase.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

CUDAProductBase::CUDAProductBase(int device, std::shared_ptr<cuda::stream_t<>> stream, std::shared_ptr<cuda::event_t> event):
  stream_(std::move(stream)),
  event_(std::move(event)),
  device_(device)
{}

bool CUDAProductBase::isAvailable() const {
  // In absence of event, the product was available already at the end
  // of produce() of the producer.
  if(not event_) {
    return true;
  }
  return event_->has_occurred();
}
