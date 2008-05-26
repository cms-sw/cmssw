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
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "Fireworks/Core/interface/FWDisplayEvent.h"

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

void MuonsProxyRhoPhiZ2DBuilder::build(const FWEventItem* iItem, 
				       TEveElementList** product, 
				       bool showEndcap,
				       bool tracksOnly)
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
   outerPropagator->SetRnrDaughters(true);
   outerPropagator->RefPMAtt().SetMarkerStyle(3);
   outerPropagator->RefPMAtt().SetMarkerColor(Color_t(kBlue));
   //units are Telsa
   double gMagneticField = FWDisplayEvent::getMagneticField();
   // double gMagneticField = 0;
   innerPropagator->SetMagField( -gMagneticField);
   double maxR = 350;
   double maxZ = 650;
   outerPropagator->SetMagField( gMagneticField * 2.5/4);
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
	   
	   // add muon segments
	   addMatchInformation( &(*muon), iItem, outerTrack, muonList, showEndcap );
	   outerTrack->MakeTrack();
	   muonList->AddElement( outerTrack );
	}

	if ( useStandAloneFit )
	  {
	     if ( ! useLastPoint ) {
		// order points with increasing radius
		if (  muon->standAloneMuon()->innerPosition().R() <  muon->standAloneMuon()->outerPosition().R() ) {
		   outerRecTrack.fP = TEveVector( muon->standAloneMuon()->innerMomentum().x(),
						  muon->standAloneMuon()->innerMomentum().y(),
						  muon->standAloneMuon()->innerMomentum().z() );
		   outerRecTrack.fV = TEveVector( muon->standAloneMuon()->innerPosition().x(), 
						  muon->standAloneMuon()->innerPosition().y(), 
						  muon->standAloneMuon()->innerPosition().z() );
		   outerRecTrack.fSign = muon->charge();
		} else { 
		   // special case (cosmics)
		   // track points inside, so we assume it points down and flip momentum sign
		   outerRecTrack.fP = TEveVector( -muon->standAloneMuon()->outerMomentum().x(),
						  -muon->standAloneMuon()->outerMomentum().y(),
						  -muon->standAloneMuon()->outerMomentum().z() );
		   outerRecTrack.fV = TEveVector( muon->standAloneMuon()->outerPosition().x(), 
						  muon->standAloneMuon()->outerPosition().y(), 
						  muon->standAloneMuon()->outerPosition().z() );
		   outerRecTrack.fSign = muon->charge();
		}
	     }
	     // printf("Initial momentum: %0.1f, %0.1f, %0.1f\n", 
	     //	    outerRecTrack.fP.fX, outerRecTrack.fP.fY, outerRecTrack.fP.fZ);
	     // printf("Initial vertex: %0.1f, %0.1f, %0.1f\n",
	     //	    outerRecTrack.fV.fX, outerRecTrack.fV.fY, outerRecTrack.fV.fZ);
		      
	     TEveTrack* outerTrack = new TEveTrack( &outerRecTrack, outerPropagator );
	     outerTrack->SetRnrPoints( true );
	     outerTrack->SetMarkerSize( 5 );
	     outerTrack->SetMainColor( iItem->defaultDisplayProperties().color() );
	     
	     TEvePathMark mark( TEvePathMark::kDaughter );
	     if (  muon->standAloneMuon()->innerPosition().R() <  muon->standAloneMuon()->outerPosition().R() ) {
		mark.fV = TEveVector( muon->standAloneMuon()->outerPosition().x(),
				      muon->standAloneMuon()->outerPosition().y(),
				      muon->standAloneMuon()->outerPosition().z() );
		mark.fTime = muon->standAloneMuon()->outerPosition().R();
	     } else {
		mark.fV = TEveVector( muon->standAloneMuon()->innerPosition().x(),
				      muon->standAloneMuon()->innerPosition().y(),
				      muon->standAloneMuon()->innerPosition().z() );
		mark.fTime = muon->standAloneMuon()->innerPosition().R();
	     }
	     outerTrack->AddPathMark( mark );
	     
	     // addHitsAsPathMarks( muon->standAloneMuon()->extra().get(), iItem->getGeom(), outerTrack);
	     
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
						      bool showEndcap,
						      bool tracksOnly)
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
	      if (! tracksOnly) parentList->AddElement(shape);
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
/*
void MuonsProxyRhoPhiZ2DBuilder::addHitsAsPathMarks( const reco::TrackExtra* recoTrack,
						     const DetIdToMatrix* geom,
						     TEveTrack* eveTrack )
{
   for ( unsigned int i = 0; i < recoTrack->recHitsSize(); ++i )
     {
	if ( recoTrack->recHit(i)->geographicalId().subdetId() != MuonSubdetId::DT ) continue;
	DTChamberId id( recoTrack->recHit(i)->geographicalId() );
	const TGeoHMatrix* matrix = geom->getMatrix( id.rawId() );
	if ( matrix ) {
	   TEvePathMark mark( TEvePathMark::kCluster2D );
	   
	   Double_t local[3];
	   local[0] = recoTrack->recHit(i)->parameters()[0];
	   local[1] = 0;
	   local[2] = 0;
	   Double_t global[3];
	   Double_t global2[3];
	   
	   matrix->LocalToMaster( local, global );
	   mark.fV = TEveVector( global[0], global[1], global[2] );

	   // printf("hit id: %d, global (x,y,z): (%0.1f, %0.1f, %0.1f), local x: %0.1f\n",
	   //	  id.rawId(), global[0], global[1], global[2], local[0] );
	   
	   local[1] = 1;
	   local[2] = 0;
	   matrix->LocalToMaster( local, global2 );
	   mark.fE = TEveVector( global2[0]-global[0], global2[1]-global[1], global2[2]-global[2] );

	   // printf("  strip (x,y,z): (%0.1f, %0.1f, %0.1f)\n",
	   //	  global2[0]-global[0], global2[1]-global[1], global2[2]-global[2] );

	   local[1] = 0;
	   local[2] = 1;
	   matrix->LocalToMaster( local, global2 );
	   mark.fP = TEveVector( global2[0]-global[0], global2[1]-global[1], global2[2]-global[2] );
	   
	   //printf("  normal (x,y,z): (%0.1f, %0.1f, %0.1f)\n",
	   // global2[0]-global[0], global2[1]-global[1], global2[2]-global[2] );
	   eveTrack->AddPathMark( mark );
	}
     }
}
*/

