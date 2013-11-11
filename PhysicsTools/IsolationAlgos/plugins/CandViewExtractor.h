#ifndef MuonIsolation_CandViewExtractor_H
#define MuonIsolation_CandViewExtractor_H

#include <string>
#include <vector>


#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

namespace muonisolation {

class CandViewExtractor : public reco::isodeposit::IsoDepositExtractor {

public:

  CandViewExtractor(){};
  CandViewExtractor(const edm::ParameterSet& par);

  virtual ~CandViewExtractor(){}

  virtual void fillVetos (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::TrackCollection & cand) { }

/*  virtual reco::IsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & cand) const;

  virtual reco::IsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & cand) const;
*/

  virtual reco::IsoDeposit deposit (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & muon) const { 
        return depositFromObject(ev, evSetup, muon);
  }

  virtual reco::IsoDeposit deposit (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & muon) const { 
        return depositFromObject(ev, evSetup, muon);
  }

private:
  reco::IsoDeposit::Veto veto( const reco::IsoDeposit::Direction & dir) const;

  template<typename T>
  reco::IsoDeposit depositFromObject( const edm::Event & ev,
      const edm::EventSetup & evSetup, const T &cand) const ;
   
  // Parameter set
  edm::InputTag theCandViewTag; // Track Collection Label
  std::string theDepositLabel;         // name for deposit
  double theDiff_r;                    // transverse distance to vertex
  double theDiff_z;                    // z distance to vertex
  double theDR_Max;                    // Maximum cone angle for deposits
  double theDR_Veto;                   // Veto cone angle
};

}

#endif
