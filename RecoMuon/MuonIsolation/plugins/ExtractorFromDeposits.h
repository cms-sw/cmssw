#ifndef MuonIsolation_ExtractorFromDeposits_H
#define MuonIsolation_ExtractorFromDeposits_H

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <string>

namespace muonisolation {

class ExtractorFromDeposits : public reco::isodeposit::IsoDepositExtractor {

public:

  ExtractorFromDeposits(){};
  ExtractorFromDeposits(const edm::ParameterSet& par);

  virtual ~ExtractorFromDeposits(){}

  virtual void fillVetos ( const edm::Event & ev, const edm::EventSetup & evSetup, 
      const reco::TrackCollection & tracks);
  virtual reco::IsoDeposit deposit (const edm::Event & ev, const edm::EventSetup & evSetup, 
      const reco::Track & track) const;
  virtual reco::IsoDeposit deposit (const edm::Event & ev, const edm::EventSetup & evSetup, 
      const reco::TrackRef & track) const;

private:
  edm::InputTag theCollectionTag;
};

}

#endif

