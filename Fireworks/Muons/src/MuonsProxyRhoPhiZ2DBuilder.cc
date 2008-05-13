#include "Fireworks/Muons/interface/MuonsProxyRhoPhiZ2DBuilder.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveManager.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
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
   TEveTrackPropagator* outerPropagator = new TEveTrackPropagator();
   //units are Telsa
   innerPropagator->SetMagField( -4.0);
   double maxR = 350;
   double maxZ = 650;
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
	// REVISE ME !
	// 
	// If we deal with tracker muons we use projected inner tracker trajectory
	// to draw the muon and position of recontructed muon segments as hits. In all
	// other cases ( stand alone muons first of all ), reco hits, segments, fit 
	// results etc are used to draw the trajectory. No hits are show, but chambers
	// with hits are visible.

	std::stringstream s;
	s << "muon" << index;
        //in order to keep muonList having the same number of elements as 'muons' we will always
        // create a list even if it will get no children
	TEveElementList* muonList = new TEveElementList(s.str().c_str());
        gEve->AddElement( muonList, tList );
	
	// need to decide how deep the inner propagator should go
	// - tracker muon (R=350, Z=650)
	// - non-tracker muon - first stand alone muon state if available
	//                    - otherwise (R=350, Z=650)
	innerPropagator->SetMaxR( maxR );
	innerPropagator->SetMaxZ( maxZ );	
	bool useStandAloneFit = ! muon->isMatchesValid() && 
	  muon->standAloneMuon().isAvailable() && 
	  muon->standAloneMuon()->extra().isAvailable();
	if ( useStandAloneFit )
	  {
	     innerPropagator->SetMaxR( muon->standAloneMuon()->innerPosition().Rho()+10 );
	     innerPropagator->SetMaxZ( fabs(muon->standAloneMuon()->innerPosition().z())+10 );
	  }
	
	Double_t lastPointVX2(0), lastPointVY2(0), lastPointVZ2(0), lastPointVX1(0), lastPointVY1(0), lastPointVZ1(0);
	bool useLastPoint = false;
	
	// draw inner track if information is available
        if ( muon->track().isAvailable() ) {
	   innerRecTrack.fP = TEveVector( muon->track()->px(), muon->track()->py(), muon->track()->pz() );
	   innerRecTrack.fV = TEveVector( muon->track()->vertex().x(), 
					  muon->track()->vertex().y(), 
					  muon->track()->vertex().z() );
	   innerRecTrack.fSign = muon->charge();
	   // std::cout << "px " << innerRecTrack.fP.fX << " py " << innerRecTrack.fP.fY << " pz " << innerRecTrack.fP.fZ
	   // << " vx " << innerRecTrack.fV.fX << " vy " << innerRecTrack.fV.fY << " vz " << innerRecTrack.fV.fZ
	   // << " sign " << innerRecTrack.fSign << std::endl;
	
	   TEveTrack* innerTrack = new TEveTrack( &innerRecTrack, innerPropagator );
	   innerTrack->SetMainColor( iItem->defaultDisplayProperties().color() );
	   // add outer most state position to guide the propagator 
	   // if the information is available
	   if ( muon->track()->extra().isAvailable() ) {
	      TEvePathMark mark( TEvePathMark::kDaughter );
	      mark.fV = TEveVector( muon->track()->outerPosition().x(),
				    muon->track()->outerPosition().y(),
				    muon->track()->outerPosition().z() );
	      innerTrack->AddPathMark( mark );
	   }
	   if ( useStandAloneFit ) {
	      TEvePathMark mark( TEvePathMark::kDaughter );
	      mark.fV = TEveVector( muon->standAloneMuon()->innerPosition().x(), 
				    muon->standAloneMuon()->innerPosition().y(), 
				    muon->standAloneMuon()->innerPosition().z() );
	      innerTrack->AddPathMark( mark );
	   }
	   innerTrack->MakeTrack();
	   
	   // get last two points of the innerTrack trajectory
	   innerTrack->GetPoint( innerTrack->GetLastPoint(),   lastPointVX2, lastPointVY2, lastPointVZ2);
	   innerTrack->GetPoint( innerTrack->GetLastPoint()-1, lastPointVX1, lastPointVY1, lastPointVZ1);
	   useLastPoint = true;
	   muonList->AddElement( innerTrack );
	}
	
	if ( useLastPoint ) {
	   outerRecTrack.fV = TEveVector(lastPointVX2,lastPointVY2,lastPointVZ2);
	   float scale = muon->p4().P()/sqrt( (lastPointVX2-lastPointVX1)*(lastPointVX2-lastPointVX1) + (lastPointVY2-lastPointVY1)*(lastPointVY2-lastPointVY1) + (lastPointVZ2-lastPointVZ1)*(lastPointVZ2-lastPointVZ1) );
	   outerRecTrack.fP = TEveVector(scale*(lastPointVX2-lastPointVX1), scale*(lastPointVY2-lastPointVY1),scale*(lastPointVZ2-lastPointVZ1));
	}

	if ( muon->isTrackerMuon() && ! useStandAloneFit ) {
	   outerRecTrack.fSign = innerRecTrack.fSign;
	   TEveTrack* outerTrack = new TEveTrack( &outerRecTrack, outerPropagator );
	   outerTrack->SetMainColor( iItem->defaultDisplayProperties().color() );
	   // std::cout << "\tpx " << outerRecTrack.fP.fX << " py " << outerRecTrack.fP.fY << " pz " << outerRecTrack.fP.fZ
	   //  << " lastPointVX " << outerRecTrack.fV.fX << " vy " << outerRecTrack.fV.fY << " vz " << outerRecTrack.fV.fZ
	   //  << " sign " << outerRecTrack.fSign << std::endl;
	   muonList->AddElement( outerTrack );
	   
	   // add muon segments
	   addMatchInformation( &(*muon), iItem, outerTrack, muonList, showEndcap );
	   outerTrack->MakeTrack();
	   muonList->AddElement( outerTrack );
	}

	if ( useStandAloneFit )
	  {
	     if ( ! useLastPoint ) {
		outerRecTrack.fP = TEveVector( muon->standAloneMuon()->innerMomentum().x(),
					       muon->standAloneMuon()->innerMomentum().y(),
					       muon->standAloneMuon()->innerMomentum().z() );
		outerRecTrack.fV = TEveVector( muon->standAloneMuon()->innerPosition().x(), 
					       muon->standAloneMuon()->innerPosition().y(), 
					       muon->standAloneMuon()->innerPosition().z() );
		outerRecTrack.fSign = muon->charge();
	     }
	     
	     TEveTrack* outerTrack = new TEveTrack( &outerRecTrack, outerPropagator );
	     outerTrack->SetMainColor( iItem->defaultDisplayProperties().color() );
	     
	     TEvePathMark mark( TEvePathMark::kDaughter );
	     mark.fV = TEveVector( muon->standAloneMuon()->outerPosition().x(),
				   muon->standAloneMuon()->outerPosition().y(),
				   muon->standAloneMuon()->outerPosition().z() );
	     outerTrack->AddPathMark( mark );
	     outerTrack->MakeTrack();
	     muonList->AddElement( outerTrack );
	  }

	
	
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

void MuonsProxyRhoPhiZ2DBuilder::addMatchInformation( const reco::Muon* muon,
						      const FWEventItem* iItem,
						      TEveTrack* track,
						      TEveElementList* parentList,
						      bool showEndcap)
{
   const DetIdToMatrix* geom = iItem->getGeom();
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
	   TEveGeoShapeExtract* extract = geom->getExtract( chamber->id.rawId() );
	   if(0!=extract) {
	      TEveElement* shape = TEveGeoShape::ImportShapeExtract(extract,0);
	      shape->IncDenyDestroy();
	      shape->SetMainTransparency(50);
	      shape->SetMainColor(iItem->defaultDisplayProperties().color());
	      parentList->AddElement(shape);
	   }
	}
	const TGeoHMatrix* matrix = geom->getMatrix( chamber->id.rawId() );
	if ( matrix ) {
	   // make muon segment 20 cm long along local z-axis
	   matrix->LocalToMaster( localTrajectoryPoint, globalTrajectoryPoint );
		   
	   // add path marks to force outer propagator to follow the expected
	   // track position
	   if ( track ) {
	      TEvePathMark mark( TEvePathMark::kDaughter );
	      mark.fV = TEveVector( globalTrajectoryPoint[0], globalTrajectoryPoint[1], globalTrajectoryPoint[2] );
	      track->AddPathMark( mark );
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
   if ( ! matches.empty() ) parentList->AddElement( segmentSet.release() );
}

