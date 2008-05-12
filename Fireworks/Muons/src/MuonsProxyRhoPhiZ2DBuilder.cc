#include "Fireworks/Muons/interface/MuonsProxyRhoPhiZ2DBuilder.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveManager.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TEveStraightLineSet.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "RVersion.h"
#include "TEveGeoNode.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "TColor.h"
#include "TEvePolygonSetProjected.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

MuonsProxyRhoPhiZ2DBuilder::MuonsProxyRhoPhiZ2DBuilder()
{
}

MuonsProxyRhoPhiZ2DBuilder::~MuonsProxyRhoPhiZ2DBuilder()
{
}

void MuonsProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem, TEveElementList** product)
{
   build(iItem, product, false);
}

void MuonsProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem, TEveElementList** product)
{
   build(iItem, product, true);
}

void MuonsProxyRhoPhiZ2DBuilder::build(const FWEventItem* iItem, TEveElementList** product, bool showEndcap)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"trackerMuons",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }
   
   // ATTN: I was not able to keep the propagators in memory, 
   //       something probably takes care of their destruction.
   //       So here they are recreated for each event.
   TEveTrackPropagator* innerPropagator = new TEveTrackPropagator();
   TEveTrackPropagator* innerShortPropagator = new TEveTrackPropagator();
   TEveTrackPropagator* outerPropagator = new TEveTrackPropagator();
   //units are Telsa
   innerPropagator->SetMagField( -4.0);
   innerPropagator->SetMaxR( 350 );
   innerPropagator->SetMaxZ( 650 );
   innerShortPropagator->SetMagField( -4.0);
   innerShortPropagator->SetMaxR( 120 );
   innerShortPropagator->SetMaxZ( 300 );
   outerPropagator->SetMagField( 2.5);
   outerPropagator->SetMaxR( 750 );
   outerPropagator->SetMaxZ( 1100 );

   const reco::MuonCollection* muons=0;
   iItem->get(muons);
   //fwlite::Handle<reco::MuonCollection> muons;
   //muons.getByLabel(*iEvent,"trackerMuons");
   
   if(0 == muons ) {
      std::cout <<"failed to get trackerMuons"<<std::endl;
      return;
   }
   
   TEveRecTrack innerRecTrack;
   TEveRecTrack outerRecTrack;
   innerRecTrack.fBeta = 1.;
   outerRecTrack.fBeta = 1.;
   unsigned int index(0);
   for ( reco::MuonCollection::const_iterator muon = muons->begin(); muon != muons->end(); ++muon, ++index )
     {
	std::stringstream s;
	s << "muon" << index;
        //in order to keep muonList having the same number of elements as 'muons' we will always
        // create a list even if it will get no children
	TEveElementList* muonList = new TEveElementList(s.str().c_str());
        gEve->AddElement( muonList, tList );
        
        //CDJ NOTE: I don't believe a proxy should ever filter the data it is representing
	innerRecTrack.fP = TEveVector( muon->p4().px(), muon->p4().py(), muon->p4().pz() );
	innerRecTrack.fV = TEveVector( muon->vertex().x(), muon->vertex().y(), muon->vertex().z() );
	innerRecTrack.fSign = muon->charge();
	// std::cout << "px " << innerRecTrack.fP.fX << " py " << innerRecTrack.fP.fY << " pz " << innerRecTrack.fP.fZ
	// << " vx " << innerRecTrack.fV.fX << " vy " << innerRecTrack.fV.fY << " vz " << innerRecTrack.fV.fZ
	// << " sign " << innerRecTrack.fSign << std::endl;
	
	TEveTrack* innerTrack = 0;
	if ( muon->numberOfMatches(reco::Muon::SegmentAndTrackArbitration) >= 2 )
	  innerTrack = new TEveTrack( &innerRecTrack, innerPropagator );
	else
	  innerTrack = new TEveTrack( &innerRecTrack, innerShortPropagator );
	
	innerTrack->SetMainColor( iItem->defaultDisplayProperties().color() );
	TEveTrack* outerTrack = 0;
	innerTrack->MakeTrack();
	muonList->AddElement( innerTrack );
           
	if ( muon->numberOfMatches(reco::Muon::SegmentAndTrackArbitration) < 2 ) continue;
        
	// get last two points of the innerTrack trajectory
	// NOTE: if RECO is available we can use the stand alone muon track 
	//       inner most state as a starting point for the outter track
	Double_t vx2, vy2, vz2, vx1, vy1, vz1;
	innerTrack->GetPoint( innerTrack->GetLastPoint(),   vx2, vy2, vz2);
	innerTrack->GetPoint( innerTrack->GetLastPoint()-1, vx1, vy1, vz1);
	   
	// second track only for barrel for now
	if ( fabs(vz2) < 650 ) {
	   outerRecTrack.fV = TEveVector(vx2,vy2,vz2);
	   // use muon momentum at IP as an estimate of its momentum at the solenoid
	   // and last two points of the inner track to get direction.
	   // NOTE: RECO can provide better estimate
	   float scale = muon->p4().P()/sqrt( (vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1) );
	   outerRecTrack.fP = TEveVector(scale*(vx2-vx1), scale*(vy2-vy1),scale*(vz2-vz1));
	   outerRecTrack.fSign = innerRecTrack.fSign;
	   outerTrack = new TEveTrack( &outerRecTrack, outerPropagator );
	   outerTrack->SetMainColor( iItem->defaultDisplayProperties().color() );
	   // std::cout << "\tpx " << outerRecTrack.fP.fX << " py " << outerRecTrack.fP.fY << " pz " << outerRecTrack.fP.fZ
	   //  << " vx " << outerRecTrack.fV.fX << " vy " << outerRecTrack.fV.fY << " vz " << outerRecTrack.fV.fZ
	   //  << " sign " << outerRecTrack.fSign << std::endl;
	   muonList->AddElement( outerTrack );
	}
           
           
	// add muon segments
	const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
	Double_t localTrajectoryPoint[3];
	Double_t globalTrajectoryPoint[3];
	//need to use auto_ptr since the segmentSet may not be passed to muonList
	std::auto_ptr<TEveStraightLineSet> segmentSet(new TEveStraightLineSet);
	segmentSet->SetLineWidth(4);
	segmentSet->SetMainColor(iItem->defaultDisplayProperties().color());

	std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin();
	for ( ; chamber != matches.end(); ++ chamber )
	  {
	     // expected track position
	     localTrajectoryPoint[0] = chamber->x;
	     localTrajectoryPoint[1] = chamber->y;
	     localTrajectoryPoint[2] = 0;
	     
	     DetId id = chamber->id;
	     if ( id.subdetId() != MuonSubdetId::CSC || showEndcap ) {
		TEveGeoShapeExtract* extract = m_item->getGeom()->getExtract( chamber->id.rawId() );
		if(0!=extract) {
		   TEveElement* shape = TEveGeoShape::ImportShapeExtract(extract,0);
		   shape->IncDenyDestroy();
		   shape->SetMainTransparency(50);
		   shape->SetMainColor(iItem->defaultDisplayProperties().color());
		   muonList->AddElement(shape);
		}
	     }
	     
	     const TGeoHMatrix* matrix = m_item->getGeom()->getMatrix( chamber->id.rawId() );
	     if ( matrix ) {
		// make muon segment 20 cm long along local z-axis
		matrix->LocalToMaster( localTrajectoryPoint, globalTrajectoryPoint );
		
		// add path marks to force outer propagator to follow the expected
		// track position
		if ( outerTrack ) {
		   TEvePathMark mark( TEvePathMark::kDaughter );
		   mark.fV = TEveVector( globalTrajectoryPoint[0], globalTrajectoryPoint[1], globalTrajectoryPoint[2] );
		   outerTrack->AddPathMark( mark );
		}
                 
		// std::cout << "\t " << " vx " << globalTrajectoryPoint[0] << " vy " << globalTrajectoryPoint[1] << 
		//  " vz " << globalTrajectoryPoint[2] <<  std::endl;
                 
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
	if ( ! matches.empty() ) muonList->AddElement( segmentSet.release() );
	if (outerTrack) outerTrack->MakeTrack();
        //}
	/*
	// adjust muon chamber visibility
	TEveElementIter iter(muonList,"\\d{8}");
	while ( TEveElement* element = iter.current() ) {
	   element->SetMainTransparency(50);
	   element->SetMainColor(Color_t(TColor::GetColor("#7f0000")));
	   if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
	     poly->SetLineColor(Color_t(TColor::GetColor("#ff0000")));
	   iter.next();
	}
         */
     }
   
}
