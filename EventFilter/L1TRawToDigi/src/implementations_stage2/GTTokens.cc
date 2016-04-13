#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "GTTokens.h"

namespace l1t {
   namespace stage2 {
      GTTokens::GTTokens(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) 
      {
         auto gttag = cfg.getParameter<edm::InputTag>("GtInputLabel");
         auto exttag = cfg.getParameter<edm::InputTag>("ExtInputLabel");
         auto calotag = cfg.getParameter<edm::InputTag>("CaloInputLabel");
         auto muontag = cfg.getParameter<edm::InputTag>("GmtInputLabel");

	 //cout << "DEBUG:  GmtInputLabel" <<  muontag << "\n";

         muonToken_ = cc.consumes<MuonBxCollection>(muontag);
	 egammaToken_ = cc.consumes<EGammaBxCollection>(calotag);
         etSumToken_ = cc.consumes<EtSumBxCollection>(calotag);
         jetToken_ = cc.consumes<JetBxCollection>(calotag);
         tauToken_ = cc.consumes<TauBxCollection>(calotag);
         algToken_ = cc.consumes<GlobalAlgBlkBxCollection>(gttag);
         extToken_ = cc.consumes<GlobalExtBlkBxCollection>(exttag);

      }
   }
}
