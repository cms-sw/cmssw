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
// $Id: Pi0Algo.h,v 1.2 2007/04/05 19:27:49 dwjang Exp $
//
//


// system include files
#include <memory>
#include "DataFormats/Common/interface/Handle.h"
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

    math::XYZTLorentzVector calculateMomentumWRT(const math::XYZTLorentzVector &momentum, const math::XYZPoint &vertex) const;

    math::XYZPoint calculatePositionAtEcal(const math::XYZTLorentzVector &momentum) const;

    void fillPi0sUsingPF(edm::Handle<reco::PFCandidateCollection> &pFCandidateHandle);

    // setters
    void setConeSize(double v) { coneSize_ = v; }

    void setUse3DAngle(bool v) { use3DAngle_ = v; }

    void setEcalEntrance(double v) { ecalEntrance_ = v; }

    void setMassRes(double v) { massRes_ = v; }

  private:

    // cone size
    double coneSize_;

    // use 3D angle (default is dR)
    bool use3DAngle_;

    // radius of ecal entrance ( this will be replaced by geometry constant later if I find it )
    double ecalEntrance_;

    // mass resolution of pi0 from two photons
    // to select pi0s based on mass cut
    double massRes_;

    // seedTrack will be a center of cone
    reco::TrackRef seedTrack_;

    // pi0 collection in cone
    reco::Pi0Collection pi0Collection_;

  };

}
#endif

