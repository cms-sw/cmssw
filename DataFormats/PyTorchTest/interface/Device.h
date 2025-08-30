#ifndef DATA_FORMATS__PYTORCH_TEST__INTERFACE__DEVICE_H_
#define DATA_FORMATS__PYTORCH_TEST__INTERFACE__DEVICE_H_

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/PyTorchTest/interface/Layout.h"

namespace torchportable {

  template <typename TDev>
  using ParticleCollectionDevice = PortableDeviceCollection<ParticleSoA, TDev>;

  template <typename TDev>
  using ClassificationCollectionDevice = PortableDeviceCollection<ClassificationSoA, TDev>;

  template <typename TDev>
  using RegressionCollectionDevice = PortableDeviceCollection<RegressionSoA, TDev>;

}  // namespace torchportable

#endif  // DATA_FORMATS__PYTORCH_TEST__INTERFACE__DEVICE_H_
