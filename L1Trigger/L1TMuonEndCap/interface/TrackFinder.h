#ifndef L1TMuonEndCap_TrackFinder_h
#define L1TMuonEndCap_TrackFinder_h

#include <memory>
#include <string>
#include <vector>
#include <array>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
  GeometryTranslator geometry_translator_;

  ConditionHelper condition_helper_;

  SectorProcessorLUT sector_processor_lut_;

  std::unique_ptr<PtAssignmentEngine> pt_assign_engine_;

  emtf::sector_array<SectorProcessor> sector_processors_;

  const edm::ParameterSet config_;

  const edm::EDGetToken tokenCSC_, tokenRPC_, tokenCPPF_, tokenGEM_;

  int verbose_, primConvLUT_;

  bool fwConfig_, useCSC_, useRPC_, useCPPF_, useGEM_;

  std::string era_;
};

#endif
