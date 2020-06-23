#ifndef L1TMuonEndCap_TrackFinder_h
#define L1TMuonEndCap_TrackFinder_h

#include <array>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"
#include "L1Trigger/L1TMuonEndCap/interface/EMTFSetup.h"
#include "L1Trigger/L1TMuonEndCap/interface/EMTFSubsystemCollector.h"
#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessor.h"

class TrackFinder {
public:
  explicit TrackFinder(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iConsumes);
  ~TrackFinder();

  void process(
      // Input
      const edm::Event& iEvent,
      const edm::EventSetup& iSetup,
      // Output
      EMTFHitCollection& out_hits,
      EMTFTrackCollection& out_tracks);

private:
  EMTFSetup setup_;

  emtf::sector_array<SectorProcessor> sector_processors_;

  // Various tokens
  const edm::EDGetToken tokenDTPhi_;
  const edm::EDGetToken tokenDTTheta_;
  const edm::EDGetToken tokenCSC_;
  const edm::EDGetToken tokenCSCComparator_;
  const edm::EDGetToken tokenRPC_;
  const edm::EDGetToken tokenCPPF_;
  const edm::EDGetToken tokenGEM_;
  const edm::EDGetToken tokenME0_;

  int verbose_;
};

#endif
