#include "RecoJets/JetProducers/interface/JetMuonHitsIDHelper.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
// #include "TrackingTools/TrackAssociator/interface/MuonDetIdAssociator.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TMath.h"
#include <vector>
#include <iostream>

using namespace std;


reco::helper::JetMuonHitsIDHelper::JetMuonHitsIDHelper( edm::ParameterSet const & pset, edm::ConsumesCollector&& iC )
{
  isRECO_ = true; // This will be "true" initially, then if the product isn't found, set to false once
  numberOfHits1RPC_ = 0;
  numberOfHits2RPC_ = 0;
  numberOfHits3RPC_ = 0;
  numberOfHits4RPC_ = 0;
  numberOfHitsRPC_ = 0;
  rpcRecHits_ = pset.getParameter<edm::InputTag>("rpcRecHits");

  input_rpchits_token_ = iC.consumes<RPCRecHitCollection>(rpcRecHits_);
 
}




void reco::helper::JetMuonHitsIDHelper::calculate( const edm::Event& event, const edm::EventSetup & iSetup, 
						   const reco::Jet &jet, const int iDbg )
{

  // initialize
  numberOfHits1RPC_ = 0;
  numberOfHits2RPC_ = 0;
  numberOfHits3RPC_ = 0;
  numberOfHits4RPC_ = 0;
  numberOfHitsRPC_ = 0;


  if ( isRECO_ ) { // This will be "true" initially, then if the product isn't found, set to false once

    // Get tracking geometry
    edm::ESHandle<GlobalTrackingGeometry> trackingGeometry;
    iSetup.get<GlobalTrackingGeometryRecord> ().get(trackingGeometry);
    
    //####READ RPC RecHits Collection########
    //#In config: RpcRecHits     = cms.InputTag("rpcRecHits")
    edm::Handle<RPCRecHitCollection> rpcRecHits_handle;
    event.getByToken(input_rpchits_token_, rpcRecHits_handle);


    if ( ! rpcRecHits_handle.isValid()  ) {
      // don't throw exception if not running on RECO
      edm::LogWarning("DataNotAvailable") << "JetMuonHitsIDHelper will not be run at all, this is not a RECO file.";
      isRECO_ = false;
      return;
    }

    //####calculate rpc  variables for each jet########
    
    for (  RPCRecHitCollection::const_iterator itRPC = rpcRecHits_handle->begin(),
	     itRPCEnd = rpcRecHits_handle->end();
	   itRPC != itRPCEnd; ++itRPC) {
      RPCRecHit const & hit = *itRPC;
      DetId detid = hit.geographicalId();
      LocalPoint lp = hit.localPosition();
      const GeomDet* gd = trackingGeometry->idToDet(detid);
      GlobalPoint gp = gd->toGlobal(lp);
      double dR2 = reco::deltaR(jet.eta(), jet.phi(), 
				static_cast<double>( gp.eta() ), static_cast<double>(gp.phi()) );
      if (dR2 < 0.5) {
	RPCDetId rpcChamberId = (RPCDetId) detid;
	numberOfHitsRPC_++;
	if (rpcChamberId.station() == 1)
	  numberOfHits1RPC_++;
	if (rpcChamberId.station() == 2)
	  numberOfHits2RPC_++;
	if (rpcChamberId.station() == 3)
	  numberOfHits3RPC_++;
	if (rpcChamberId.station() == 4)
	  numberOfHits4RPC_++;
      }
    }
  }
}
