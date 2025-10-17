#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CaloLayer1Tokens.h"

namespace l1t {
  namespace stage2 {
    CaloLayer1Tokens::CaloLayer1Tokens(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) {
      auto ecalTag = cfg.getParameter<edm::InputTag>("ecalDigis");
      auto hcalTag = cfg.getParameter<edm::InputTag>("hcalDigis");
      auto regionTag = cfg.getParameter<edm::InputTag>("caloRegions");
      auto cicadaTag = cfg.getParameter<edm::InputTag>("CICADAScore");

      ecalDigiToken_ = cc.consumes<EcalTrigPrimDigiCollection>(ecalTag);
      hcalDigiToken_ = cc.consumes<HcalTrigPrimDigiCollection>(hcalTag);
      caloRegionToken_ = cc.consumes<L1CaloRegionCollection>(regionTag);
      cicadaToken_ = cc.consumes<CICADABxCollection>(cicadaTag);
    }
  }  // namespace stage2
}  // namespace l1t
