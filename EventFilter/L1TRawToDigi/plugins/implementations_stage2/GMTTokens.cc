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
         auto imdBmtfTag = cfg.getParameter<edm::InputTag>("ImdInputLabelBMTF");
         auto imdEmtfNegTag = cfg.getParameter<edm::InputTag>("ImdInputLabelEMTFNeg");
         auto imdEmtfPosTag = cfg.getParameter<edm::InputTag>("ImdInputLabelEMTFPos");
         auto imdOmtfNegTag = cfg.getParameter<edm::InputTag>("ImdInputLabelOMTFNeg");
         auto imdOmtfPosTag = cfg.getParameter<edm::InputTag>("ImdInputLabelOMTFPos");

         regionalMuonCandTokenBMTF_ = cc.consumes<RegionalMuonCandBxCollection>(bmtfTag);
         regionalMuonCandTokenOMTF_ = cc.consumes<RegionalMuonCandBxCollection>(omtfTag);
         regionalMuonCandTokenEMTF_ = cc.consumes<RegionalMuonCandBxCollection>(emtfTag);
         muonToken_ = cc.consumes<MuonBxCollection>(tag);
         imdMuonTokenBMTF_ = cc.consumes<MuonBxCollection>(imdBmtfTag);
         imdMuonTokenEMTFNeg_ = cc.consumes<MuonBxCollection>(imdEmtfNegTag);
         imdMuonTokenEMTFPos_ = cc.consumes<MuonBxCollection>(imdEmtfPosTag);
         imdMuonTokenOMTFNeg_ = cc.consumes<MuonBxCollection>(imdOmtfNegTag);
         imdMuonTokenOMTFPos_ = cc.consumes<MuonBxCollection>(imdOmtfPosTag);
      }
   }
}
