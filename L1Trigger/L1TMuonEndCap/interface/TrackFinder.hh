#ifndef L1TMuonEndCap_TrackFinder_hh
#define L1TMuonEndCap_TrackFinder_hh

#include <memory>
#include <string>
#include <vector>
#include <array>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessorLUT.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.hh"
#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessor.hh"


class TrackFinder {
public:
  explicit TrackFinder(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iConsumes);
  ~TrackFinder();

  void resetPtLUT(std::shared_ptr<const L1TMuonEndCapForest> ptLUT);

  void process(
      // Input
      const edm::Event& iEvent, const edm::EventSetup& iSetup,
      // Output
      EMTFHitCollection& out_hits,
      EMTFTrackCollection& out_tracks
  ) const;

private:
  // 'mutable' because GeometryTranslator has to 'update' inside the const function
  mutable GeometryTranslator geometry_translator_;

  SectorProcessorLUT sector_processor_lut_;

  PtAssignmentEngine pt_assign_engine_;

  sector_array<SectorProcessor> sector_processors_;

  const edm::ParameterSet config_;

  const edm::EDGetToken tokenCSC_, tokenRPC_;

  int verbose_;

  bool useCSC_, useRPC_;
};

#endif
