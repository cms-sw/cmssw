#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/PyTorchAllocatorBridge.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/PyTorchAlpakaService.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  PyTorchAlpakaService::PyTorchAlpakaService(edm::ParameterSet const&, edm::ActivityRegistry&) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    edm::LogInfo("PyTorchAlpakaService") << "Plugging CMSSW's device caching allocator into PyTorch.";
    PyTorchAllocatorBridge::install();
#endif
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_SERVICE(ALPAKA_TYPE_ALIAS(PyTorchAlpakaService));
