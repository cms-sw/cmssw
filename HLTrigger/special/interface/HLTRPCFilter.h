#ifndef HLTRPCFilter_h
#define HLTRPCFilter_h

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

//
// class declaration
//

class HLTRPCFilter : public edm::global::EDFilter<> {
   public:
      explicit HLTRPCFilter(const edm::ParameterSet&);
      ~HLTRPCFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      virtual bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

      edm::EDGetTokenT<RPCRecHitCollection> rpcRecHitsToken;
      edm::EDGetTokenT<RPCRecHitCollection> rpcDTPointsToken;
      edm::EDGetTokenT<RPCRecHitCollection> rpcCSCPointsToken;
      double rangestrips;
};

#endif
