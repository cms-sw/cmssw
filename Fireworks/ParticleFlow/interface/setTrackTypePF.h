#ifndef Fireworks_ParticleFlow_setTrackTypePF_h
#define Fireworks_ParticleFlow_setTrackTypePF_h
// -*- C++ -*-
//

// system include files
#include "Rtypes.h"
#include "TAttLine.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

// user include files

// forward declarations
namespace reco {
  class PFCandidate;
}
class TEveTrack;

namespace fireworks {
  void setTrackTypePF(const reco::PFCandidate& pfCand, TAttLine* track);

}

#endif
