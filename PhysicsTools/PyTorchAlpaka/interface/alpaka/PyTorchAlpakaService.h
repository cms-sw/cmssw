#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchAlpakaService_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_PyTorchAlpakaService_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace edm {
  class ActivityRegistry;
  class ParameterSet;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PyTorchAlpakaService {
  public:
    PyTorchAlpakaService(edm::ParameterSet const&, edm::ActivityRegistry&);
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DECLARE_ALPAKA_TYPE_ALIAS(PyTorchAlpakaService);
#endif
