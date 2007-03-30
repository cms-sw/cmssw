#ifndef MuonIsolation_ExtractorFromDeposits_H
#define MuonIsolation_ExtractorFromDeposits_H

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <string>

namespace muonisolation {

class ExtractorFromDeposits : public MuIsoExtractor {

public:

  ExtractorFromDeposits(){};
  ExtractorFromDeposits(const edm::ParameterSet& par);

  virtual ~ExtractorFromDeposits(){}

  virtual void fillVetos ( const edm::Event & ev, const edm::EventSetup & evSetup, 
      const reco::TrackCollection & tracks);
  virtual reco::MuIsoDeposit deposit (const edm::Event & ev, const edm::EventSetup & evSetup, 
      const reco::Track & track) const;
  virtual reco::MuIsoDeposit deposit (const edm::Event & ev, const edm::EventSetup & evSetup, 
      const reco::TrackRef & track) const;

private:
  edm::InputTag theCollectionTag;
};

}

#endif

