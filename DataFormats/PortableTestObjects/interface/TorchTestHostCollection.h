#ifndef DataFormats_PortableTestObjects_interface_TorchTestHostCollection_h
#define DataFormats_PortableTestObjects_interface_TorchTestHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TorchTestSoA.h"

namespace torchportabletest {

  using ParticleHostCollection = PortableHostCollection<ParticleSoA>;
  using SimpleNetHostCollection = PortableHostCollection<SimpleNetSoA>;
  using MultiHeadNetHostCollection = PortableHostCollection<MultiHeadNetSoA>;
  using ImageHostCollection = PortableHostCollection<Image>;
  using LogitsHostCollection = PortableHostCollection<Logits>;

}  // namespace torchportabletest

#endif  // DataFormats_PortableTestObjects_interface_TorchTestHostCollection_h
