#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CaloTokens.h"

namespace l1t {
   namespace stage1 {
      CaloTokens::CaloTokens(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) : PackerTokens(cfg, cc)
      {
         auto tag = cfg.getParameter<edm::InputTag>("InputLabel");
         auto tautag = cfg.getParameter<edm::InputTag>("TauInputLabel");
         auto isotautag = cfg.getParameter<edm::InputTag>("IsoTauInputLabel");
         auto tagHFBitCounts = cfg.getParameter<edm::InputTag>("HFBitCountsInputLabel");
         auto tagHFRingSums = cfg.getParameter<edm::InputTag>("HFRingSumsInputLabel");
         auto tagRegion = cfg.getParameter<edm::InputTag>("RegionInputLabel");
         auto tagEmCand = cfg.getParameter<edm::InputTag>("EmCandInputLabel");

         towerToken_ = cc.consumes<CaloTowerBxCollection>(tag);
         egammaToken_ = cc.consumes<EGammaBxCollection>(tag);
         etSumToken_ = cc.consumes<EtSumBxCollection>(tag);
         jetToken_ = cc.consumes<JetBxCollection>(tag);
         tauToken_ = cc.consumes<TauBxCollection>(tautag);
         isotauToken_ = cc.consumes<TauBxCollection>(isotautag);
         calospareHFBitCountsToken_ = cc.consumes<CaloSpareBxCollection>(tagHFBitCounts);
         calospareHFRingSumsToken_ = cc.consumes<CaloSpareBxCollection>(tagHFRingSums);
         caloregionToken_ = cc.consumes<L1CaloRegionCollection>(tagRegion);
         caloemCandToken_ = cc.consumes<L1CaloEmCollection>(tagEmCand);

         
      }
   }
}
