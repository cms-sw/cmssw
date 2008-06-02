#ifndef MuonIsolation_CandViewExtractor_H
#define MuonIsolation_CandViewExtractor_H

#include <string>
#include <vector>


#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"

namespace muonisolation {

class CandViewExtractor : public MuIsoExtractor {

public:

  CandViewExtractor(){};
  CandViewExtractor(const edm::ParameterSet& par);

  virtual ~CandViewExtractor(){}

  virtual void fillVetos (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::TrackCollection & cand) { }

/*  virtual reco::MuIsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & cand) const;

  virtual reco::MuIsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & cand) const;
*/

  virtual reco::MuIsoDeposit deposit (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & muon) const { 
        return depositFromObject(ev, evSetup, muon);
  }

  virtual reco::MuIsoDeposit deposit (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & muon) const { 
        return depositFromObject(ev, evSetup, muon);
  }

private:
  reco::MuIsoDeposit::Veto veto( const reco::MuIsoDeposit::Direction & dir) const;

  template<typename T>
  reco::MuIsoDeposit depositFromObject( const edm::Event & ev,
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
