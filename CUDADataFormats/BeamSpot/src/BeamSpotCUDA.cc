#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

BeamSpotCUDA::BeamSpotCUDA(Data const* data_h, cuda::stream_t<>& stream) {
  edm::Service<CUDAService> cs;

  data_d_ = cs->make_device_unique<Data>(stream);
  cuda::memory::async::copy(data_d_.get(), data_h, sizeof(Data), stream.id());
}
