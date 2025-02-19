#ifndef HLTRPCFilter_h
#define HLTRPCFilter_h

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

//
// class declaration
//

class HLTRPCFilter : public edm::EDFilter {
   public:
      explicit HLTRPCFilter(const edm::ParameterSet&);
      ~HLTRPCFilter();

   private:
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      double rangestrips;
      edm::InputTag rpcRecHitsLabel;
      edm::InputTag rpcDTPointsLabel;
      edm::InputTag rpcCSCPointsLabel;

      
      // ----------member data ---------------------------
};

#endif
