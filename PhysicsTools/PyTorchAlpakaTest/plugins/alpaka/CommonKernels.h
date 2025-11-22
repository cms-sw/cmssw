#ifndef PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_CommonKernels_h
#define PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_CommonKernels_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/MaskDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest::kernels {

  void randomFillParticleCollection(Queue& queue, portabletest::ParticleDeviceCollection& particles);
  void randomFillImageCollection(Queue& queue, portabletest::ImageDeviceCollection& images);
  void fillMask(Queue& queue, portabletest::MaskDeviceCollection& mask);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest::kernels

#endif  // PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_CommonKernels_h
