#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "GTTokens.h"

namespace l1t {
   namespace stage2 {
      GTTokens::GTTokens(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) : PackerTokens(cfg, cc)
      {
         auto tag = cfg.getParameter<edm::InputTag>("InputLabel");

	 egammaToken_ = cc.consumes<EGammaBxCollection>(tag);
         etSumToken_ = cc.consumes<EtSumBxCollection>(tag);
         jetToken_ = cc.consumes<JetBxCollection>(tag);
         tauToken_ = cc.consumes<TauBxCollection>(tag);
         algToken_ = cc.consumes<GlobalAlgBlkBxCollection>(tag);
         extToken_ = cc.consumes<GlobalExtBlkBxCollection>(tag);

      }
   }
}
