#ifndef PhysicsTools_ONNXRuntimeAlpakaTest_plugins_alpaka_CommonKernels_h
#define PhysicsTools_ONNXRuntimeAlpakaTest_plugins_alpaka_CommonKernels_h

#include "DataFormats/PortableTestObjects/interface/alpaka/HitDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/MaskDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest::kernels {

  // fill pt, eta and phi with random values in [0, 1)
  void randomFillParticleCollection(Queue& queue, portabletest::ParticleDeviceCollection& particles, uint32_t seed);
  // fill the r, g and b channels with independent random values in [0, 1)
  void randomFillImageCollection(Queue& queue, portabletest::ImageDeviceCollection& images, uint32_t seed);
  // fill x, y and z with random values in [0, 1), and set the hitScale scalar to 1.5
  void randomFillHitCollection(Queue& queue, portabletest::HitDeviceCollection& hits, uint32_t seed);
  // mask the eta feature only
  void fillMask(Queue& queue, portabletest::MaskDeviceCollection& mask);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest::kernels

#endif  // PhysicsTools_ONNXRuntimeAlpakaTest_plugins_alpaka_CommonKernels_h
