#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "GMTTokens.h"

namespace l1t {
   namespace stage2 {
      GMTTokens::GMTTokens(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) 
      {
         auto bmtfTag = cfg.getParameter<edm::InputTag>("BMTFInputLabel");
         auto omtfTag = cfg.getParameter<edm::InputTag>("OMTFInputLabel");
         auto emtfTag = cfg.getParameter<edm::InputTag>("EMTFInputLabel");
         auto tag = cfg.getParameter<edm::InputTag>("InputLabel");

         regionalMuonCandTokenBMTF_ = cc.consumes<RegionalMuonCandBxCollection>(bmtfTag);
         regionalMuonCandTokenOMTF_ = cc.consumes<RegionalMuonCandBxCollection>(omtfTag);
         regionalMuonCandTokenEMTF_ = cc.consumes<RegionalMuonCandBxCollection>(emtfTag);
         muonToken_ = cc.consumes<MuonBxCollection>(tag);
      }
   }
}
