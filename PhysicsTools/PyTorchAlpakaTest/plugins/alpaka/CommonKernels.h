#ifndef PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_CommonKernels_h
#define PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_CommonKernels_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/MaskDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest::kernels {

  using namespace portabletest;

  void randomFillParticleCollection(Queue& queue, ParticleDeviceCollection& particles);
  void randomFillImageCollection(Queue& queue, ImageDeviceCollection& images);
  void fillMask(Queue& queue, MaskDeviceCollection& mask);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest::kernels

#endif  // PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_CommonKernels_h
