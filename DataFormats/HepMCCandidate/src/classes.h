#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/PxPyPzE4D.h" 
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    edm::Wrapper<std::vector<reco::GenParticleCandidate> > w1;
  }
}
