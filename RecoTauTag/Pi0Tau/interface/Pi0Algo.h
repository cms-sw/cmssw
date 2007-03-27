#ifndef RecoTauTag_Pi0Tau_Pi0Algo_h_
#define RecoTauTag_Pi0Tau_Pi0Algo_h_

// -*- C++ -*-
//
// Package:    Pi0Algo
// Class:      Pi0Algo
// 
/**\class Pi0Algo Pi0Algo.h RecoTauTag/Pi0Tau/interface/Pi0Algo.h

 Description: algorithms to reconstruct Pi0

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dongwook Jang
//         Created:  Tue Jan  9 16:40:36 CST 2007
// $Id$
//
//


// system include files
#include <memory>
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "RecoTauTag/Pi0Tau/interface/Pi0.h"
#include "RecoTauTag/Pi0Tau/interface/Pi0Fwd.h"

//
// class decleration
//

namespace reco {

  class Pi0Algo {

  public:

    Pi0Algo(reco::TrackRef seedTrack);

    ~Pi0Algo();

    const reco::Pi0Collection &pi0Collection() const { return pi0Collection_; }

    void fillPi0sUsingPF(edm::Handle<reco::PFCandidateCollection> &pFCandidateHandle, double coneSize=0.524);

    math::XYZTLorentzVector calculateMomentumWRT(const math::XYZTLorentzVector &momentum, const math::XYZPoint &vertex) const;
    math::XYZPoint calculatePositionAtEcal(const math::XYZTLorentzVector &momentum) const;

  private:

    // cone size
    double coneSize_;

    // seedTrack will be a center of cone
    reco::TrackRef seedTrack_;

    // pi0 collection in cone
    reco::Pi0Collection pi0Collection_;

  };

}
#endif

