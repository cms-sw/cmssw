#ifndef CommonToolsParticleFlow_PDCandWithSuperCluster_H
#define CommonToolsParticleFlow_PDCandWithSuperCluster_H

#include <string>
#include <vector>


#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"


class PFCandWithSuperClusterExtractor : public reco::isodeposit::IsoDepositExtractor {

public:

  PFCandWithSuperClusterExtractor(){};
  PFCandWithSuperClusterExtractor(const edm::ParameterSet& par);

  virtual ~PFCandWithSuperClusterExtractor(){}

  virtual void fillVetos (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::TrackCollection & cand) { }


  virtual reco::IsoDeposit deposit (const edm::Event & ev,
				    const edm::EventSetup & evSetup, const reco::Track & muon) const { 
    return depositFromObject(ev, evSetup, muon);
  }
  
  virtual reco::IsoDeposit deposit (const edm::Event & ev,
				    const edm::EventSetup & evSetup, const reco::Candidate & photon) const { 

    const reco::Photon * myPhoton= dynamic_cast<const reco::Photon*>(&photon);
    if(myPhoton)
      return depositFromObject(ev, evSetup,*myPhoton);
    
    const reco::PFCandidate * myPFCand = dynamic_cast<const reco::PFCandidate*>(&photon);
    return depositFromObject(ev, evSetup,*myPFCand);
  }

private:
  reco::IsoDeposit::Veto veto( const reco::IsoDeposit::Direction & dir) const;

  reco::IsoDeposit depositFromObject( const edm::Event & ev,
				      const edm::EventSetup & evSetup, const reco::Photon &cand) const ;

  reco::IsoDeposit depositFromObject( const edm::Event & ev,
				      const edm::EventSetup & evSetup, const reco::Track &cand) const ;

  reco::IsoDeposit depositFromObject( const edm::Event & ev,
				      const edm::EventSetup & evSetup, const reco::PFCandidate &cand) const ;

  // Parameter set
  edm::InputTag thePFCandTag; // Track Collection Label
  std::string theDepositLabel;         // name for deposit
  bool theVetoSuperClusterMatch;         //SuperClusterRef Check
  double theDiff_r;                    // transverse distance to vertex
  double theDiff_z;                    // z distance to vertex
  double theDR_Max;                    // Maximum cone angle for deposits
  double theDR_Veto;                   // Veto cone angle
};


#endif
