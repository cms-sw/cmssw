#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

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

class RPCPointProducer : public edm::stream::EDProducer<> {
public:
  explicit RPCPointProducer(const edm::ParameterSet&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<CSCSegmentCollection> cscSegments;
  edm::EDGetTokenT<DTRecSegment4DCollection> dt4DSegments;
  edm::EDGetTokenT<reco::TrackCollection> tracks;

  std::unique_ptr<DTSegtoRPC> dtSegtoRPC;
  std::unique_ptr<CSCSegtoRPC> cscSegtoRPC;
  std::unique_ptr<TracktoRPC> tracktoRPC;

  const bool incldt;
  const bool inclcsc;
  const bool incltrack;
  const bool debug;
  const double MinCosAng;
  const double MaxD;
  const double MaxDrb4;
  const double ExtrapolatedRegion;
};
