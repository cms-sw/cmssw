#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EMTFTokens.h"

namespace l1t {
  namespace stage2 {
    EMTFTokens::EMTFTokens(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) {
      auto tag = cfg.getParameter<edm::InputTag>("InputLabel");

      regionalMuonCandToken_ = cc.consumes<RegionalMuonCandBxCollection>(tag);
      regionalMuonShowerToken_ = cc.consumes<RegionalMuonShowerBxCollection>(tag);
      EMTFDaqOutToken_ = cc.consumes<EMTFDaqOutCollection>(tag);
      EMTFHitToken_ = cc.consumes<EMTFHitCollection>(tag);
      EMTFTrackToken_ = cc.consumes<EMTFTrackCollection>(tag);
      EMTFLCTToken_ = cc.consumes<CSCCorrelatedLCTDigiCollection>(tag);
      EMTFCSCShowerToken_ = cc.consumes<CSCShowerDigiCollection>(tag);
      EMTFCPPFToken_ = cc.consumes<CPPFDigiCollection>(tag);
      EMTFGEMPadClusterToken_ = cc.consumes<GEMPadDigiClusterCollection>(tag);
    }
  }  // namespace stage2
}  // namespace l1t
