#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "RecoLocalMuon/RPCRecHit/interface/DTSegtoRPC.h"
#include "RecoLocalMuon/RPCRecHit/interface/CSCSegtoRPC.h"
#include "RecoLocalMuon/RPCRecHit/interface/TracktoRPC.h" 

//
// class decleration
//

class RPCPointProducer : public edm::global::EDProducer<> {
   public:
      explicit RPCPointProducer(const edm::ParameterSet&);

   private:
      virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

      const edm::EDGetTokenT<CSCSegmentCollection> cscSegments;
      const edm::EDGetTokenT<DTRecSegment4DCollection> dt4DSegments;
      const edm::EDGetTokenT<reco::TrackCollection> tracks;
      const edm::InputTag tracks_;

      const bool incldt;
      const bool inclcsc;
      const bool incltrack; 
      const bool debug;
      const double MinCosAng;
      const double MaxD;
      const double MaxDrb4;
      const double ExtrapolatedRegion;
      const edm::ParameterSet trackTransformerParam;
};

