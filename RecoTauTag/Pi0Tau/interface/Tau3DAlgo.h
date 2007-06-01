#ifndef RecoTauTag_Pi0Tau_Tau3DAlgo_h_
#define RecoTauTag_Pi0Tau_Tau3DAlgo_h_

// -*- C++ -*-
//
// Package:    Tau3DAlgo
// Class:      Tau3DAlgo
// 
/**\class Tau3DAlgo Tau3DAlgo.h RecoTauTag/Pi0Tau/interface/Tau3DAlgo.h

 Description: to calculate tau variables for Tau3D

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dongwook Jang
//         Created:  Tue Jan  9 16:40:36 CST 2007
// $Id: Tau3DAlgo.h,v 1.2 2007/04/05 19:27:49 dwjang Exp $
//
//


// system include files
#include <memory>
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoTauTag/Pi0Tau/interface/Pi0.h"
#include "RecoTauTag/Pi0Tau/interface/Pi0Fwd.h"
#include "RecoTauTag/Pi0Tau/interface/Tau3D.h"
#include "RecoTauTag/Pi0Tau/interface/Tau3DFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <vector>

//
// class decleration
//

namespace reco {

  class Tau3DAlgo {

  public:
    
    Tau3DAlgo(edm::Handle<reco::TrackCollection> *trackHandle);

    ~Tau3DAlgo();

    const reco::Tau3DCollection &tau3DCollection() const { return tau3DCollection_; }

    void setUse3DAngle(bool v) { use3DAngle_ = v; }

    void setTauOuterConeSize(double v) { tauOuterConeSize_ = v; }

    void setSeedTrackThreshold(double v) { seedTrackPtThreshold_ = v; }

    void fillTau3Ds(edm::Handle<reco::PFCandidateCollection> &pFCandidateHandle);

    void fillRefVectors();

    void findSeedTracks();

  private:

    // isolation cone size of tau (default is 0.524)
    double tauOuterConeSize_;

    // seed track threshold (default is 5 GeV)
    double seedTrackPtThreshold_;

    // flag to use 3D angle (default is dR)
    bool   use3DAngle_;

    edm::Handle<reco::TrackCollection> *trackHandle_;

    std::vector<reco::TrackRef> trackRefs_;
    std::vector<reco::TrackRef> seedTrackCandidateRefs_;

    std::vector<reco::TrackRef> seedTrackRefs_;

    reco::Tau3DCollection tau3DCollection_;

  };

}
#endif

