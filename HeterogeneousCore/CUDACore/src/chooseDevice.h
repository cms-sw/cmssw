#ifndef HeterogeneousCore_CUDACore_chooseDevice_h
#define HeterogeneousCore_CUDACore_chooseDevice_h

#include "FWCore/Utilities/interface/StreamID.h"

namespace cms::cuda {
  int chooseDevice(edm::StreamID id);
}

#endif
