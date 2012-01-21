#ifndef HLTRPCFilter_h
#define HLTRPCFilter_h

// user include files

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

//
// class declaration
//

class HLTRPCFilter : public HLTFilter {
   public:
      explicit HLTRPCFilter(const edm::ParameterSet&);
      ~HLTRPCFilter();

   private:
      virtual void beginJob() ;
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
      virtual void endJob() ;
      double rangestrips;
      edm::InputTag rpcRecHitsLabel;
      edm::InputTag rpcDTPointsLabel;
      edm::InputTag rpcCSCPointsLabel;

      
      // ----------member data ---------------------------
};

#endif
