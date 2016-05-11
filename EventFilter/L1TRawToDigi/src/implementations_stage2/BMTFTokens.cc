#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "BMTFTokens.h"

namespace l1t {
   namespace stage2 {
      BMTFTokens::BMTFTokens(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) 
      {
         auto ouputTag = cfg.getParameter<edm::InputTag>("InputLabel");
         auto inputTagPh = cfg.getParameter<edm::InputTag>("InputLabel");
         auto inputTagTh = cfg.getParameter<edm::InputTag>("InputLabel");
         
				 outputMuonToken_ = cc.consumes<RegionalMuonCandBxCollection>(ouputTag);
				 inputMuonTokenPh_ = cc.consumes<L1MuDTChambPhContainer>(inputTagPh);
				 inputMuonTokenTh_ = cc.consumes<L1MuDTChambThContainer>(inputTagTh);
         
      }
   }
}
