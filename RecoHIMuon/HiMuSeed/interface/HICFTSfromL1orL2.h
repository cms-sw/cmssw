#ifndef HICFTSfromL1orL2_H
#define HICFTSfromL1orL2_H

//CommonDet
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

// Muon trigger

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

// L2MuonReco

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "MagneticField/Engine/interface/MagneticField.h"

//CLHEP includes
#include <CLHEP/Vector/LorentzVector.h>
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Vector/ThreeVector.h"
#include <cmath>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>


//-----------------------------------------------------------------------------
namespace cms {
class HICFTSfromL1orL2
{

  public:

    HICFTSfromL1orL2(const MagneticField * mf){field = mf;}
    virtual ~HICFTSfromL1orL2(){}
    std::vector<FreeTrajectoryState> createFTSfromL1(std::vector<L1MuGMTExtendedCand>&);
    std::vector<FreeTrajectoryState> createFTSfromL2(const reco::RecoChargedCandidateCollection& rc);
    std::vector<FreeTrajectoryState> createFTSfromStandAlone(const reco::TrackCollection& rc);
    std::vector<FreeTrajectoryState> createFTSfromL1orL2(std::vector<L1MuGMTExtendedCand>& gmt, const reco::RecoChargedCandidateCollection& recmuons);
    
  private:
    FreeTrajectoryState FTSfromL1(const L1MuGMTExtendedCand& gmt);
    FreeTrajectoryState FTSfromL2(const reco::RecoChargedCandidate& gmt);
    FreeTrajectoryState FTSfromStandAlone(const reco::Track& gmt);
    
    const MagneticField * field;
};
}
#endif

