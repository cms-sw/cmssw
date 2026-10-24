#ifndef PhysicsTools_ONNXRuntimeAlpakaTest_plugins_alpaka_CommonKernels_h
#define PhysicsTools_ONNXRuntimeAlpakaTest_plugins_alpaka_CommonKernels_h

#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/MaskDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest::kernels {

  // fill pt, eta and phi with random values in [0, 1)
  void randomFillParticleCollection(Queue& queue, portabletest::ParticleDeviceCollection& particles);
  // fill the r, g and b channels with the same random values in [0, 1)
  void randomFillImageCollection(Queue& queue, portabletest::ImageDeviceCollection& images);
  // mask the eta feature only
  void fillMask(Queue& queue, portabletest::MaskDeviceCollection& mask);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest::kernels

#endif  // PhysicsTools_ONNXRuntimeAlpakaTest_plugins_alpaka_CommonKernels_h
