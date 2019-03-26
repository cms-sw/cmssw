#ifndef HeterogeneousCore_CUDACore_chooseCUDADevice_h
#define HeterogeneousCore_CUDACore_chooseCUDADevice_h

#include "FWCore/Utilities/interface/StreamID.h"

namespace cudacore {
  int chooseCUDADevice(edm::StreamID id);
}

#endif
