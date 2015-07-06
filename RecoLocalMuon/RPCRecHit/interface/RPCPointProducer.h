#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "RecoLocalMuon/RPCRecHit/interface/DTSegtoRPC.h"
#include "RecoLocalMuon/RPCRecHit/interface/CSCSegtoRPC.h"
#include "RecoLocalMuon/RPCRecHit/interface/TracktoRPC.h" 

//
// class decleration
//

class RPCPointProducer : public edm::stream::EDProducer<> {
   public:
      explicit RPCPointProducer(const edm::ParameterSet&);
      ~RPCPointProducer();

      const edm::EDGetTokenT<CSCSegmentCollection> cscSegments;
      const edm::EDGetTokenT<DTRecSegment4DCollection> dt4DSegments;
      const edm::EDGetTokenT<reco::TrackCollection> tracks;
      const edm::InputTag tracks_;
   private:
      void beginRun(edm::Run const&, edm::EventSetup const&) override;
      void endRun(edm::Run const&, edm::EventSetup const&) override;
      void produce(edm::Event&, const edm::EventSetup&) override;
      const bool debug;
      const bool incldt;
      const bool inclcsc;
      const bool incltrack;
      const double MinCosAng;
      const double MaxD;
      const double MaxDrb4;
      const double ExtrapolatedRegion;
      const edm::ParameterSet serviceParameters;
      const edm::ParameterSet trackTransformerParam;

      // ----------member data ---------------------------
    
    ObjectMapCSC* TheCSCObjectsMap_;
    ObjectMap*    TheDTObjectsMap_;
    ObjectMap2*       TheDTtrackObjectsMap_;
    ObjectMap2CSC*    TheCSCtrackObjectsMap_;
    edm::ESWatcher<MuonGeometryRecord> MuonGeometryWatcher;
};

