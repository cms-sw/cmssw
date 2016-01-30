#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EMTFTokens.h"

namespace l1t {
  namespace stage2 {
    EMTFTokens::EMTFTokens(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) : PackerTokens(cfg, cc)
    {
      auto tag = cfg.getParameter<edm::InputTag>("InputLabel");
      
      // EMTFMuonCandToken_ = cc.consumes<EMTFMuonCandBxCollection>(tag); // Does this need a tag? which one? - AWB 11.01.15
      EMTFOutputToken_   = cc.consumes<EMTFOutputCollection>(tag);     // Does this need a tag? which one? - AWB 11.01.15
    }
  }
}
