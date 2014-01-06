#ifndef MuonIdentification_MuonIdTruthInfo_h
#define MuonIdentification_MuonIdTruthInfo_h 1

// add MC hits to a list of matched segments. The only
// way to differentiat hits is the error on the local
// hit position. It's -9999 for a MC hit
// Since it's debugging mode - code is slow

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSegmentMatch.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class MuonIdTruthInfo
{
 public:

   void registerConsumes(edm::ConsumesCollector& iC);


   static void truthMatchMuon( const edm::Event& iEvent,
			       const edm::EventSetup& iSetup,
			       reco::Muon& aMuon);
 private:
   static void checkSimHitForBestMatch(reco::MuonSegmentMatch& segmentMatch,
				       double& distance,
				       const PSimHit& hit, 
				       const DetId& chamberId,
				       const edm::ESHandle<GlobalTrackingGeometry>& geometry);
   
   static double matchChi2(    const reco::Track& recoTrk,
			       const SimTrack& simTrk);
};
#endif
