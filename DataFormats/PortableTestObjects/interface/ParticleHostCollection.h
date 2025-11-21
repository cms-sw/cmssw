#ifndef DataFormats_PortableTestObjects_interface_ParticleHostCollection_h
#define DataFormats_PortableTestObjects_interface_ParticleHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ParticleSoA.h"

namespace portabletest {

  using ParticleHostCollection = PortableHostCollection<ParticleSoA>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_ParticleHostCollection_h
