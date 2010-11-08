// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonBuilder
// $Id: FWMuonBuilder.cc,v 1.15 2010/01/21 21:02:13 amraktad Exp $
//

// system include files
#include "TEveTrackPropagator.h"
#include "TEveVSDStructs.h"
#include "TEveCompound.h"
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveStraightLineSet.h"
#include "TEveGeoNode.h"

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Muons/interface/FWMuonBuilder.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"


namespace  {
std::vector<TEveVector> getRecoTrajectoryPoints( const reco::Muon* muon,
                                                 const FWEventItem* iItem )
{
   std::vector<TEveVector> points;
   const DetIdToMatrix* geom = iItem->getGeom();
   const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
   Double_t localTrajectoryPoint[3];
   Double_t globalTrajectoryPoint[3];
   std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin();
   for ( ; chamber != matches.end(); ++chamber )
   {
      // expected track position
      localTrajectoryPoint[0] = chamber->x;
      localTrajectoryPoint[1] = chamber->y;
      localTrajectoryPoint[2] = 0;

      DetId id = chamber->id;
      const TGeoHMatrix* matrix = geom->getMatrix( chamber->id.rawId() );
      if ( matrix ) {
         matrix->LocalToMaster( localTrajectoryPoint, globalTrajectoryPoint );
         points.push_back(TEveVector(globalTrajectoryPoint[0],
                                     globalTrajectoryPoint[1],
                                     globalTrajectoryPoint[2]));
      }
   }
   return points;
}

//______________________________________________________________________________

void addMatchInformation( const reco::Muon* muon,
                          const FWEventItem* iItem,
                          TEveElement* parentList,
                          bool showEndcap)
{
   std::set<unsigned int> ids;
   const DetIdToMatrix* geom = iItem->getGeom();
   const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
   //need to use auto_ptr since the segmentSet may not be passed to muonList
   std::auto_ptr<TEveStraightLineSet> segmentSet(new TEveStraightLineSet);
   segmentSet->SetLineWidth(4);
   segmentSet->SetMainColor(iItem->defaultDisplayProperties().color());
   std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin();
   for ( ; chamber != matches.end(); ++chamber )
   {
      DetId id = chamber->id;
      if ( ids.insert(id.rawId()).second &&  // ensure that we add same chamber only once
           ( id.subdetId() != MuonSubdetId::CSC || showEndcap ) ){
         TEveGeoShape* shape = geom->getShape( chamber->id.rawId() );
         if(0!=shape) {
            shape->RefMainTrans().Scale(0.999, 0.999, 0.999);
            shape->SetMainTransparency(65);
            shape->SetMainColor(iItem->defaultDisplayProperties().color());
            parentList->AddElement(shape);
         }
      }
      const TGeoHMatrix* matrix = geom->getMatrix( chamber->id.rawId() );
      if ( matrix ) {
         // make muon segment 20 cm long along local z-axis
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
   if ( !matches.empty() ) parentList->AddElement( segmentSet.release() );
}

//______________________________________________________________________________

bool
buggyMuon( const reco::Muon* muon,
           const DetIdToMatrix* geom )
{
   if (!muon->standAloneMuon().isAvailable() ||
       !muon->standAloneMuon()->extra().isAvailable() ) return false;
   const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
   Double_t localTrajectoryPoint[3];
   Double_t globalTrajectoryPoint[3];
   std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin();
   for ( ; chamber != matches.end(); ++chamber )
   {
      // expected track position
      localTrajectoryPoint[0] = chamber->x;
      localTrajectoryPoint[1] = chamber->y;
      localTrajectoryPoint[2] = 0;

      DetId id = chamber->id;
      const TGeoHMatrix* matrix = geom->getMatrix( chamber->id.rawId() );
      if ( matrix ) {
         matrix->LocalToMaster( localTrajectoryPoint, globalTrajectoryPoint );
         double phi = atan2(globalTrajectoryPoint[1],globalTrajectoryPoint[0]);
         if ( cos( phi - muon->standAloneMuon()->innerPosition().phi()) < 0 )
            return true;
      }
   }
   return false;
}

}

//
// constructors and destructor
//
FWMuonBuilder::FWMuonBuilder()
{
   //NOTE: We call IncRefCount and IncDenyDestroy since TEveTrackPropagator actually has two reference counts being done on it
   // We only want the one using IncRefCount to actually cause the deletion which is why 'IncDenyDestroy' does not have a matching
   // DecDenyDestroy.  I'm still using a edm::FWEvePtr to hold the Propagator since I want to know if the propagator is deleted
   m_trackerPropagator.reset(new TEveTrackPropagator()); // propagate within tracker
   m_trackerPropagator->IncRefCount();
   m_trackerPropagator->IncDenyDestroy();
   m_trackerPropagator->SetMaxR( 850 );
   m_trackerPropagator->SetMaxZ( 1100 );
   m_trackerPropagator->SetMaxStep(5);
}

FWMuonBuilder::~FWMuonBuilder()
{
   m_trackerPropagator->DecRefCount();
}

//
// member functions
//
//______________________________________________________________________________

void
FWMuonBuilder::calculateField(const reco::Muon& iData, FWMagField* field)
{

   // if auto field estimation mode, do extra loop over muons.
   // use both inner and outer track if available
   if ( field->getAutodetect() ) {
     if ( fabs( iData.eta() ) > 2.0 || iData.pt() < 3 ) return;
     if ( iData.innerTrack().isAvailable() ){
       double estimate = fw::estimate_field(*(iData.innerTrack()),true);
       if ( estimate >= 0 ) field->guessField( estimate );
	 
     }
     if ( iData.outerTrack().isAvailable() ){
       double estimate = fw::estimate_field(*(iData.outerTrack()));
       if ( estimate >= 0 ) field->guessFieldIsOn( estimate > 0.5 );
     }
   }
}

//______________________________________________________________________________

void
FWMuonBuilder::buildMuon(const FWEventItem* iItem,
                         const reco::Muon* muon,
                         TEveElement* tList,
                         bool showEndcap,
                         bool tracksOnly)
{
   calculateField(*muon, iItem->context().getField());

   // workaround for missing GetFieldObj() in TEveTrackPropagator, default stepper is kHelix
   if (m_trackerPropagator->GetStepper() == TEveTrackPropagator::kHelix) {
      m_trackerPropagator->SetStepper(TEveTrackPropagator::kRungeKutta);
      m_trackerPropagator->SetMagFieldObj(iItem->context().getField());
   }

   TEveRecTrack recTrack;
   recTrack.fBeta = 1.;

   // If we deal with a tracker muon we use the inner track and guide it
   // through the trajectory points from the reconstruction. Segments
   // represent hits. Matching between hits and the trajectory shows
   // how well the inner track matches with the muon hypothesis.
   //
   // In other cases we use a global muon track with a few states from 
   // the inner and outer tracks or just the outer track if it's the
   // only option

   if ( muon->isTrackerMuon() && 
	muon->innerTrack().isAvailable() &&
	muon->isMatchesValid() &&
	!buggyMuon( &*muon, iItem->getGeom() ) )
   {
      TEveTrack* trk = fireworks::prepareTrack(*(muon->innerTrack()),
                                               m_trackerPropagator.get(),
                                               iItem->defaultDisplayProperties().color(),
                                               getRecoTrajectoryPoints(muon,iItem) );
      trk->MakeTrack();
      tList->AddElement( trk );
      if ( ! tracksOnly )
	 addMatchInformation( &(*muon), iItem, tList, showEndcap );
      return;
   } 

   if ( muon->isGlobalMuon() &&
	muon->globalTrack().isAvailable() )
   {
      std::vector<TEveVector> extraPoints;
      if ( muon->innerTrack().isAvailable() ){
	 extraPoints.push_back( TEveVector(muon->innerTrack()->innerPosition().x(),
					   muon->innerTrack()->innerPosition().y(),
					   muon->innerTrack()->innerPosition().z()) );
	 extraPoints.push_back( TEveVector(muon->innerTrack()->outerPosition().x(),
					   muon->innerTrack()->outerPosition().y(),
					   muon->innerTrack()->outerPosition().z()) );
      }
      if ( muon->outerTrack().isAvailable() ){
	 extraPoints.push_back( TEveVector(muon->outerTrack()->innerPosition().x(),
					   muon->outerTrack()->innerPosition().y(),
					   muon->outerTrack()->innerPosition().z()) );
	 extraPoints.push_back( TEveVector(muon->outerTrack()->outerPosition().x(),
					   muon->outerTrack()->outerPosition().y(),
					   muon->outerTrack()->outerPosition().z()) );
      }
      TEveTrack* trk = fireworks::prepareTrack(*(muon->globalTrack()),
                                               m_trackerPropagator.get(),
                                               iItem->defaultDisplayProperties().color(),
                                               extraPoints);
      trk->MakeTrack();
      tList->AddElement( trk );
      return;
   }

   if ( muon->innerTrack().isAvailable() )
   {
      TEveTrack* trk = fireworks::prepareTrack(*(muon->innerTrack()),
                                               m_trackerPropagator.get(),
                                               iItem->defaultDisplayProperties().color());
      trk->MakeTrack();
      tList->AddElement( trk );
      return;
   }

   if ( muon->outerTrack().isAvailable() )
   {
      TEveTrack* trk = fireworks::prepareTrack(*(muon->outerTrack()),
                                               m_trackerPropagator.get(),
                                               iItem->defaultDisplayProperties().color());
      trk->MakeTrack();
      tList->AddElement( trk );
      return;
   }
   
   // if got that far it means we have nothing but a candidate
   // show it anyway.
   TEveTrack* trk = fireworks::prepareTrack(*muon,
					    m_trackerPropagator.get(),
					    iItem->defaultDisplayProperties().color());
   trk->MakeTrack();
   tList->AddElement( trk );
}
