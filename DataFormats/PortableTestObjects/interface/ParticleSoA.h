#ifndef DataFormats_PortableTestObjects_interface_ParticleSoA_h
#define DataFormats_PortableTestObjects_interface_ParticleSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace portabletest {

  GENERATE_SOA_LAYOUT(ParticleLayout, SOA_COLUMN(float, pt), SOA_COLUMN(float, eta), SOA_COLUMN(float, phi))
  using ParticleSoA = ParticleLayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_ParticleSoA_h
