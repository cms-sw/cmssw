#include "Fireworks/Muons/interface/MuonsProxy3DBuilder.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveManager.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "TEveStraightLineSet.h"

MuonsProxy3DBuilder::MuonsProxy3DBuilder()
{
   // ATTN: this should be made configurable
   const char* geomtryFile = "cmsGeom10.root";
   detIdToMatrix_.loadGeometry( geomtryFile );
   detIdToMatrix_.loadMap( geomtryFile );
}

MuonsProxy3DBuilder::~MuonsProxy3DBuilder()
{
}

void MuonsProxy3DBuilder::build(const fwlite::Event* iEvent, TEveElementList** oList)
{
   if(0 == *oList) {
      TEveElementList* tlist =  new TEveElementList("Muons","trackerMuons",true);
      *oList = tlist;
      (*oList)->SetMainColor(Color_t(kRed));
      
      gEve->AddElement(*oList);
   } else {
      (*oList)->DestroyElements();
   }
   
   // ATTN: I was not able to keep the propagators in memory, 
   //       something probably takes care of their destruction.
   //       So here they are recreated for each event.
   TEveTrackPropagator* innerPropagator = new TEveTrackPropagator();
   TEveTrackPropagator* outerPropagator = new TEveTrackPropagator();
   //units are kG
   innerPropagator->SetMagField( -4.0*10.);
   innerPropagator->SetMaxR( 350 );
   innerPropagator->SetMaxZ( 650 );
   outerPropagator->SetMagField( 2.5*10.);
   outerPropagator->SetMaxR( 750 );
   outerPropagator->SetMaxZ( 1100 );

   fwlite::Handle<reco::MuonCollection> muons;
   muons.getByLabel(*iEvent,"trackerMuons");
   
   if(0 == muons.ptr() ) {
      std::cout <<"failed to get trackerMuons"<<std::endl;
      return;
   }
   
   TEveRecTrack innerRecTrack;
   TEveRecTrack outerRecTrack;
   innerRecTrack.beta = 1.;
   outerRecTrack.beta = 1.;
   unsigned int index(0);
   for ( reco::MuonCollection::const_iterator muon = muons->begin(); muon != muons->end(); ++muon, ++index )
     {
	innerRecTrack.P = TEveVector( muon->p4().px(), muon->p4().py(), muon->p4().pz() );
	innerRecTrack.V = TEveVector( muon->vertex().x(), muon->vertex().y(), muon->vertex().z() );
	innerRecTrack.sign = muon->charge();
	
	std::cout << "px " << innerRecTrack.P.x << " py " << innerRecTrack.P.y << " pz " << innerRecTrack.P.z
	  << " vx " << innerRecTrack.V.x << " vy " << innerRecTrack.V.y << " vz " << innerRecTrack.V.z
	  << " sign " << innerRecTrack.sign << std::endl;
	   
	std::stringstream s;
	s << "muon" << index;
	TEveElementList* muonList = new TEveElementList(s.str().c_str());
	   
	TEveTrack* innerTrack = new TEveTrack( &innerRecTrack, innerPropagator );
	innerTrack->SetMainColor( (*oList)->GetMainColor() );
	TEveTrack* outerTrack = 0;
	innerTrack->MakeTrack();
	muonList->AddElement( innerTrack );
	
	// get last two points of the innerTrack trajectory
	// NOTE: if RECO is available we can use the stand alone muon track 
	//       inner most state as a starting point for the outter track
	Double_t vx2, vy2, vz2, vx1, vy1, vz1;
	innerTrack->GetPoint( innerTrack->GetLastPoint(),   vx2, vy2, vz2);
	innerTrack->GetPoint( innerTrack->GetLastPoint()-1, vx1, vy1, vz1);
	   
	// second track only for barrel for now
	if ( fabs(vz2) < 650 ) {
	   outerRecTrack.V = TEveVector(vx2,vy2,vz2);
	   // use muon momentum at IP as an estimate of its momentum at the solenoid
	   // and last two points of the inner track to get direction.
	   // NOTE: RECO can provide better estimate
	   float scale = muon->p4().P()/sqrt( (vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1) );
	   outerRecTrack.P = TEveVector(scale*(vx2-vx1), scale*(vy2-vy1),scale*(vz2-vz1));
	   outerRecTrack.sign = innerRecTrack.sign;
	   outerTrack = new TEveTrack( &outerRecTrack, outerPropagator );
	   outerTrack->SetMainColor( (*oList)->GetMainColor() );
	   std::cout << "\tpx " << outerRecTrack.P.x << " py " << outerRecTrack.P.y << " pz " << outerRecTrack.P.z
	     << " vx " << outerRecTrack.V.x << " vy " << outerRecTrack.V.y << " vz " << outerRecTrack.V.z
	     << " sign " << outerRecTrack.sign << std::endl;
	   muonList->AddElement( outerTrack );
	}
   
	// add muon segments
	const std::vector<reco::MuonChamberMatch>& matches = muon->getMatches();
	Double_t localTrajectoryPoint[3];
	Double_t globalTrajectoryPoint[3];
	TEveStraightLineSet* segmentSet = new TEveStraightLineSet;
	std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin();
	for ( ; chamber != matches.end(); ++ chamber )
	  {
	     // expected track position
	     localTrajectoryPoint[0] = chamber->x;
	     localTrajectoryPoint[1] = chamber->y;
	     localTrajectoryPoint[2] = 0;
	     
	     DetId id = chamber->id;
	     const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix( chamber->id.rawId() );
	     if ( matrix ) {
		// make muon segment 20 cm long along local z-axis
		matrix->LocalToMaster( localTrajectoryPoint, globalTrajectoryPoint );
		
		// add path marks to force outer propagator to follow the expected
		// track position
		if ( outerTrack ) {
		   TEvePathMark* mark = new TEvePathMark( TEvePathMark::Daughter );
		   mark->V = TEveVector( globalTrajectoryPoint[0], globalTrajectoryPoint[1], globalTrajectoryPoint[2] );
		   outerTrack->AddPathMark( mark );
		}
		
		std::cout << "\t " << " vx " << globalTrajectoryPoint[0] << " vy " << globalTrajectoryPoint[1] << 
		  " vz " << globalTrajectoryPoint[2] <<  std::endl;

		// add segments
		for ( std::vector<reco::MuonSegmentMatch>::const_iterator segment = chamber->segmentMatches.begin();
		      segment != chamber->segmentMatches.end(); ++segment )
		  {
		     Double_t localSegmentInnerPoint[3];
		     Double_t localSegmentOuterPoint[3];
		     Double_t globalSegmentInnerPoint[3];
		     Double_t globalSegmentOuterPoint[3];
		     localSegmentOuterPoint[0] = segment->x + segment->dXdZ * 10;
		     localSegmentOuterPoint[1] = segment->y + segment->dYdZ * 10;
		     localSegmentOuterPoint[2] = 10;
		     localSegmentInnerPoint[0] = segment->x - segment->dXdZ * 10;
		     localSegmentInnerPoint[1] = segment->y - segment->dYdZ * 10;
		     localSegmentInnerPoint[2] = -10;
		     matrix->LocalToMaster( localSegmentInnerPoint, globalSegmentInnerPoint );
		     matrix->LocalToMaster( localSegmentOuterPoint, globalSegmentOuterPoint );

		     segmentSet->AddLine(globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2], 
					 globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
		  }
	     }
	  }
	if ( ! matches.empty() ) muonList->AddElement( segmentSet );
	gEve->AddElement( muonList, *oList );
	if (outerTrack) outerTrack->MakeTrack();
     }
}
