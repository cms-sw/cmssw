#ifndef MuonIsolation_ExtractorFromDeposits_H
#define MuonIsolation_ExtractorFromDeposits_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <string>

namespace muonisolation {

class ExtractorFromDeposits : public reco::isodeposit::IsoDepositExtractor {

public:

  ExtractorFromDeposits(){};
  ExtractorFromDeposits(const edm::ParameterSet& par, edm::ConsumesCollector && iC);

  ~ExtractorFromDeposits() override{}

  void fillVetos ( const edm::Event & ev, const edm::EventSetup & evSetup,
      const reco::TrackCollection & tracks) override;
  reco::IsoDeposit deposit (const edm::Event & ev, const edm::EventSetup & evSetup,
      const reco::Track & track) const override;
  virtual reco::IsoDeposit deposit (const edm::Event & ev, const edm::EventSetup & evSetup,
      const reco::TrackRef & track) const;

private:
  edm::EDGetTokenT<reco::IsoDepositMap> theCollectionToken;
};

}

#endif

