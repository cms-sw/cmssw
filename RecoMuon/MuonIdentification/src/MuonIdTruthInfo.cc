#include "RecoMuon/MuonIdentification/interface/MuonIdTruthInfo.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

void MuonIdTruthInfo::registerConsumes(edm::ConsumesCollector& iC) {
  iC.mayConsume<edm::SimTrackContainer>(edm::InputTag("g4SimHits",""));
  iC.mayConsume<edm::PSimHitContainer>(edm::InputTag("g4SimHits","MuonDTHits"));
  iC.mayConsume<edm::PSimHitContainer>(edm::InputTag("g4SimHits","MuonCSCHits"));

}



void MuonIdTruthInfo::truthMatchMuon(const edm::Event& iEvent, 
				     const edm::EventSetup& iSetup,
				     reco::Muon& aMuon)
{
   // get a list of simulated track and find a track with the best match to
   // the muon.track(). Use its id and chamber id to localize hits
   // If a hit has non-zero local z coordinate, it's position wrt
   // to the center of a chamber is extrapolated by a straight line
   
   edm::Handle<edm::SimTrackContainer> simTracks;
   iEvent.getByLabel<edm::SimTrackContainer>("g4SimHits", "", simTracks);
   if (! simTracks.isValid() ) {
      LogTrace("MuonIdentification") <<"No tracks found";
      return;
   }
   
   // get the tracking Geometry
   edm::ESHandle<GlobalTrackingGeometry> geometry;
   iSetup.get<GlobalTrackingGeometryRecord>().get(geometry);
   if ( ! geometry.isValid() )
     throw cms::Exception("FatalError") << "Unable to find GlobalTrackingGeometryRecord in event!\n";
   
   float bestMatchChi2 = 9999; //minimization creteria
   unsigned int bestMatch = 0;
   const unsigned int offset = 0; // kludge to fix a problem in trackId matching between tracks and hits.
   
   for( edm::SimTrackContainer::const_iterator simTrk = simTracks->begin();
	simTrk != simTracks->end(); simTrk++ )
     {
	float chi2 = matchChi2(*aMuon.track().get(),*simTrk);
	if (chi2>bestMatchChi2) continue;
	bestMatchChi2 = chi2;
	bestMatch = simTrk->trackId();
     }
   
   bestMatch -= offset;
   
   std::vector<reco::MuonChamberMatch>& matches = aMuon.matches();
   int numberOfTruthMatchedChambers = 0;

   // loop over chambers
   for(std::vector<reco::MuonChamberMatch>::iterator chamberMatch = matches.begin();
       chamberMatch != matches.end(); chamberMatch ++)
     {
	if (chamberMatch->id.det() != DetId::Muon ) {
	   edm::LogWarning("MuonIdentification") << "Detector id of a muon chamber corresponds to not a muon detector";
	   continue;
	}
	reco::MuonSegmentMatch bestSegmentMatch;
	double distance = 99999;
	
	if ( chamberMatch->id.subdetId() == MuonSubdetId::DT) {
	   DTChamberId detId(chamberMatch->id.rawId());

	   edm::Handle<edm::PSimHitContainer> simHits;
	   iEvent.getByLabel("g4SimHits", "MuonDTHits", simHits);
	   if ( simHits.isValid() ) {
	      for( edm::PSimHitContainer::const_iterator hit = simHits->begin(); hit != simHits->end(); hit++)
		if ( hit->trackId() == bestMatch ) checkSimHitForBestMatch(bestSegmentMatch, distance, *hit, detId, geometry );
	   }else LogTrace("MuonIdentification") <<"No DT simulated hits are found";
	}

	if ( chamberMatch->id.subdetId() == MuonSubdetId::CSC) {
	   CSCDetId detId(chamberMatch->id.rawId());

	   edm::Handle<edm::PSimHitContainer> simHits;
	   iEvent.getByLabel("g4SimHits", "MuonCSCHits", simHits);
	   if ( simHits.isValid() ) {
	      for( edm::PSimHitContainer::const_iterator hit = simHits->begin(); hit != simHits->end(); hit++)
		if ( hit->trackId() == bestMatch ) checkSimHitForBestMatch(bestSegmentMatch, distance, *hit, detId, geometry );
	   }else LogTrace("MuonIdentification") <<"No CSC simulated hits are found";
	}
	if (distance < 9999) {
	   chamberMatch->truthMatches.push_back( bestSegmentMatch );
	   numberOfTruthMatchedChambers++;
	   LogTrace("MuonIdentification") << "Best truth matched hit:" <<
	      "\tDetId: " << chamberMatch->id.rawId() << "\n" <<
	      "\tprojection: ( " << bestSegmentMatch.x << ", " << bestSegmentMatch.y << " )\n";
	}
     }
   LogTrace("MuonIdentification") << "Truth matching summary:\n\tnumber of chambers: " << matches.size() <<
     "\n\tnumber of truth matched chambers: " << numberOfTruthMatchedChambers << "\n";
}

void MuonIdTruthInfo::checkSimHitForBestMatch(reco::MuonSegmentMatch& segmentMatch,
					      double& distance,
					      const PSimHit& hit, 
					      const DetId& chamberId,
					      const edm::ESHandle<GlobalTrackingGeometry>& geometry)
{
  printf("DONT FORGET TO CALL REGISTERCONSUMES()\n");

   // find the hit position projection at the reference surface of the chamber:
   // first get entry and exit point of the hit in the global coordinates, then
   // get local coordinates of these points wrt the chamber and then find the
   // projected X-Y coordinates
   
   const GeomDet* chamberGeometry = geometry->idToDet( chamberId );
   const GeomDet* simUnitGeometry = geometry->idToDet( DetId(hit.detUnitId()) );
   
   if (chamberGeometry && simUnitGeometry ) {
      LocalPoint entryPoint = chamberGeometry->toLocal( simUnitGeometry->toGlobal( hit.entryPoint() ) );
      LocalPoint exitPoint =  chamberGeometry->toLocal( simUnitGeometry->toGlobal( hit.exitPoint()  ) );
      LocalVector direction = exitPoint - entryPoint;
      if ( fabs(direction.z()) > 0.001) {
	 LocalPoint projection = entryPoint - direction*(entryPoint.z()/direction.z());
	 if ( fabs(projection.z()) > 0.001 ) 
	   edm::LogWarning("MuonIdentification") << "z coordinate of the hit projection must be zero and it's not!\n";
	 
	 double new_distance = 99999;
	 if( entryPoint.z()*exitPoint.z() < -1 ) // changed sign, so the reference point is inside
	   new_distance = 0;
	 else {
	    if ( fabs(entryPoint.z()) < fabs(exitPoint.z()) )
	      new_distance = fabs(entryPoint.z());
	    else
	      new_distance = fabs(exitPoint.z());
	 }
	 
	 if (new_distance < distance) { 
	    // find a SimHit closer to the reference surface, update segmentMatch
	    segmentMatch.x = projection.x();
	    segmentMatch.y = projection.y();
	    segmentMatch.xErr = 0;
	    segmentMatch.yErr = 0;
	    segmentMatch.dXdZ = direction.x()/direction.z();
	    segmentMatch.dYdZ = direction.y()/direction.z();
	    segmentMatch.dXdZErr = 0;
	    segmentMatch.dYdZErr = 0;
	    distance = new_distance;
	    LogTrace("MuonIdentificationVerbose") << "Better truth matched segment found:\n" <<
	      "\tDetId: " << chamberId.rawId() << "\n" <<
	      "\tentry point: ( " << entryPoint.x() << ", " << entryPoint.y() << ", " << entryPoint.z() << " )\n" <<
	      "\texit point: ( " << exitPoint.x() << ", " << exitPoint.y() << ", " << exitPoint.z() << " )\n" <<
	      "\tprojection: ( " << projection.x() << ", " << projection.y() << ", " << projection.z() << " )\n";
	 }
      }
   }else{
      if ( ! chamberGeometry ) edm::LogWarning("MuonIdentification") << "Cannot get chamber geomtry for DetId: " << chamberId.rawId();
      if ( ! simUnitGeometry ) edm::LogWarning("MuonIdentification") << "Cannot get detector unit geomtry for DetId: " << hit.detUnitId();
   }
}

double MuonIdTruthInfo::matchChi2( const reco::Track& recoTrk, const SimTrack& simTrk)
{
   double deltaPhi = fabs(recoTrk.phi()-simTrk.momentum().phi());
   if (deltaPhi>1.8*3.1416) deltaPhi = 2*3.1416-deltaPhi; // take care of phi discontinuity
   return pow((recoTrk.p()-simTrk.momentum().rho())/simTrk.momentum().rho(),2)+
     pow(deltaPhi/(2*3.1416),2)+pow((recoTrk.theta()-simTrk.momentum().theta())/3.1416,2);
}
