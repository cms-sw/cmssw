#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/PxPyPzE4D.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    reco::PFCandidateRef c_r;
    reco::PFCandidateRefProd c_rp;
    reco::PFCandidateRefVector c_rv;
    edm::Wrapper<std::vector<reco::PFCandidate> > w1;
  }
}
