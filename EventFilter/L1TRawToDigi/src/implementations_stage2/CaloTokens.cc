#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CaloTokens.h"

namespace l1t {
   namespace stage2 {
      CaloTokens::CaloTokens(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) 
      {
         auto towtag = cfg.getParameter<edm::InputTag>("TowerInputLabel");
         auto tag = cfg.getParameter<edm::InputTag>("InputLabel");
         towerToken_ = cc.consumes<CaloTowerBxCollection>(towtag);
         egammaToken_ = cc.consumes<EGammaBxCollection>(tag);
         etSumToken_ = cc.consumes<EtSumBxCollection>(tag);
         jetToken_ = cc.consumes<JetBxCollection>(tag);
         tauToken_ = cc.consumes<TauBxCollection>(tag);
      }
   }
}
