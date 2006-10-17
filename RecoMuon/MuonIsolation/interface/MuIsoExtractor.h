#ifndef MuonIsolation_Extractor_H
#define MuonIsolation_Extractor_H

#include <vector>
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"

namespace muonisolation { class Direction; }

class MuIsoExtractor {
public:
  virtual ~MuIsoExtractor(){}
  virtual std::vector<reco::MuIsoDeposit> deposits( const edm::Event & ev, 
      const edm::EventSetup & evSetup, 
      const reco::Track & track, 
      const std::vector<muonisolation::Direction> & vetoDirs, double coneSize) const = 0;
};
#endif
