#ifndef MuonIsolation_MuIsoExtractor_H
#define MuonIsolation_MuIsoExtractor_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
                                                                                
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

namespace muonisolation {

class MuIsoExtractor {
public:
  virtual ~MuIsoExtractor(){};
  virtual void fillVetos(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & tracks) = 0;
  virtual reco::MuIsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const = 0;
};

}
#endif
