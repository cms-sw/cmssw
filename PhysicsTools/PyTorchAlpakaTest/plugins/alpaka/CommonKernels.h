#ifndef PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_CommonKernels_h
#define PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_CommonKernels_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/alpaka/MaskDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest::kernels {

  using namespace torchportabletest;

  void randomFillParticleCollection(Queue& queue, ParticleDeviceCollection& particles);
  void randomFillImageCollection(Queue& queue, ImageDeviceCollection& images);
  void fillMask(Queue& queue, MaskDevice& mask);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest::kernels

#endif  // PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_CommonKernels_h