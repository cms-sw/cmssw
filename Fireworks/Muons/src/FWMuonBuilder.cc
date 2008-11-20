// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 19 16:12:27 EST 2008
// $Id: FWMuonBuilder.cc,v 1.1 2008/11/20 01:11:05 chrjones Exp $
//

// system include files
#include "TEveTrackPropagator.h"
#include "TEveCompound.h"
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveStraightLineSet.h"
#include "TEveGeoNode.h"

// user include files
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "Fireworks/Muons/interface/FWMuonBuilder.h"

#include "Fireworks/Core/interface/prepareTrack.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
namespace  {
   void addMatchInformation( const reco::Muon* muon,
                            const FWEventItem* iItem,
                            TEveTrack* track,
                            TEveElementList* parentList,
                            bool showEndcap,
                            bool tracksOnly=false)
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
            TEveGeoShape* shape = geom->getShape( chamber->id.rawId() );
            if(0!=shape) {
               shape->SetMainTransparency(75);
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
               if ( mark.fV.Mag() > track->GetVertex().Mag() ) // avoid zigzags
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
   
   bool buggyMuon( const reco::Muon* muon,
                  const DetIdToMatrix* geom )
   {
      if (! muon->standAloneMuon().isAvailable() ||
          ! muon->standAloneMuon()->extra().isAvailable() ) return false;
      const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
      Double_t localTrajectoryPoint[3];
      Double_t globalTrajectoryPoint[3];
      std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin();
      for ( ; chamber != matches.end(); ++ chamber )
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
  
   
   TEveVector firstMatch( const reco::Muon* muon,
                         const FWEventItem* iItem )
   {
      const DetIdToMatrix* geom = iItem->getGeom();
      const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
      Double_t localTrajectoryPoint[3];
      Double_t globalTrajectoryPoint[3];
      std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin();
      for ( ; chamber != matches.end(); ++ chamber )
      {
         // expected track position
         localTrajectoryPoint[0] = chamber->x;
         localTrajectoryPoint[1] = chamber->y;
         localTrajectoryPoint[2] = 0;
         
         DetId id = chamber->id;
         const TGeoHMatrix* matrix = geom->getMatrix( chamber->id.rawId() );
         if ( matrix ) {
            matrix->LocalToMaster( localTrajectoryPoint, globalTrajectoryPoint );
            return TEveVector( globalTrajectoryPoint[0], globalTrajectoryPoint[1], globalTrajectoryPoint[2] );
         }
      }
      return TEveVector();
   }
   
   
   TEveVector muonLocation( const reco::Muon* muon,
                           const FWEventItem* iItem )
   {
      
      // stand alone information
      if (muon->standAloneMuon().isAvailable() &&
          muon->standAloneMuon()->extra().isAvailable() )
         return TEveVector ( muon->standAloneMuon()->innerPosition().x(),
                            muon->standAloneMuon()->innerPosition().y(),
                            muon->standAloneMuon()->innerPosition().z() );
      // tracker muon info
      TEveVector v = firstMatch( muon, iItem );
      if (v.Mag()>0)
         return v;
      else
         //wild guess
         return TEveVector ( muon->px(),  muon->py(),  muon->pz() );
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
   m_innerPropagator.reset(new TEveTrackPropagator()); // propagate to muon volume
   m_innerPropagator->IncRefCount();
   m_innerPropagator->IncDenyDestroy();
   m_outerPropagator.reset(new TEveTrackPropagator()); // outer muon propagator
   m_outerPropagator->IncRefCount();
   m_outerPropagator->IncDenyDestroy();
   //units are Telsa
   m_magneticField = CmsShowMain::getMagneticField();
   m_innerPropagator->SetMagField( -m_magneticField);
   m_innerPropagator->SetMaxR( 450 );
   m_innerPropagator->SetMaxZ( 750 );
   m_trackerPropagator->SetMagField( -m_magneticField);
   m_trackerPropagator->SetMaxR( 123 );
   m_trackerPropagator->SetMaxZ( 300 );
   m_outerPropagator->SetMagField( m_magneticField * 1.5/4);
   m_outerPropagator->SetMaxR( 850 );
   m_outerPropagator->SetMaxZ( 1100 );   
}

// FWMuonBuilder::FWMuonBuilder(const FWMuonBuilder& rhs)
// {
//    // do actual copying here;
// }

FWMuonBuilder::~FWMuonBuilder()
{
   m_trackerPropagator->DecRefCount();
   m_innerPropagator->DecRefCount();
   m_outerPropagator->DecRefCount();
   
}

//
// assignment operators
//
// const FWMuonBuilder& FWMuonBuilder::operator=(const FWMuonBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWMuonBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWMuonBuilder::buildMuon(const FWEventItem* iItem,
                         const reco::Muon* muon,
                         TEveElementList* tList,
                         const fw::NamedCounter& counter,
                         bool showEndcap,
                         bool tracksOnly)
{
   if(m_magneticField != CmsShowMain::getMagneticField()) {
      m_magneticField = CmsShowMain::getMagneticField();
      m_innerPropagator->SetMagField( -m_magneticField);
      m_trackerPropagator->SetMagField( -m_magneticField);
      m_outerPropagator->SetMagField( m_magneticField * 1.5/4);
   }
   
   TEveRecTrack innerRecTrack;
   TEveRecTrack outerRecTrack;
   innerRecTrack.fBeta = 1.;
   outerRecTrack.fBeta = 1.;
   // If we deal with tracker muons we use projected inner tracker trajectory
   // to draw the muon and position of recontructed muon segments as hits. In all
   // other cases ( stand alone muons first of all ), reco hits, segments, fit
   // results etc are used to draw the trajectory. No hits are show, but chambers
   // with hits are visible.
   
   //in order to keep muonList having the same number of elements as 'muons' we will always
   // create a list even if it will get no children
   const unsigned int nBuffer = 1024;
   char title[nBuffer];
   snprintf(title, nBuffer,"Muon %d, Pt: %0.1f GeV",counter.index(),muon->pt());
   TEveCompound* muonList = new TEveCompound(counter.str().c_str(), title);
   muonList->OpenCompound();
   //guarantees that CloseCompound will be called no matter what happens
   boost::shared_ptr<TEveCompound> sentry(muonList,boost::mem_fn(&TEveCompound::CloseCompound));
   gEve->AddElement( muonList, tList );
   muonList->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
   muonList->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
   
   bool useStandAloneFit = ! muon->isMatchesValid() &&
   muon->standAloneMuon().isAvailable() &&
   muon->standAloneMuon()->extra().isAvailable();
   
   if ( ! useStandAloneFit && buggyMuon( &*muon, iItem->getGeom() ) )
      useStandAloneFit = true;
   
   Double_t lastPointVX2(0), lastPointVY2(0), lastPointVZ2(0), lastPointVX1(0), lastPointVY1(0), lastPointVZ1(0);
   bool useLastPoint = false;
   bool outerTrackIsInitialized = false;
   
   // draw inner track if information is available
   if ( muon->track().isAvailable() ) {
      TEveVector location = muonLocation(&*muon, iItem);
      TEveTrack* trk = fireworks::prepareTrack(*(muon->track()),
                                               0,
                                               muonList,
                                               iItem->defaultDisplayProperties().color() );
      // if track points away from us we use its initial point as
      // the origin of the outer track with flipped momentum
      // and propagate to the exit state (decay)
      if ( location.fX*trk->GetMomentum().fX + location.fY*trk->GetMomentum().fY < 0 ) {
	 trk->SetPropagator( m_trackerPropagator.get() );
	 trk->MakeTrack();
	 muonList->AddElement( trk );
         
	 if ( muon->track()->extra().isAvailable() ) {
	    outerRecTrack.fP = TEveVector( -muon->track()->innerMomentum().x(),
                                          -muon->track()->innerMomentum().y(),
                                          -muon->track()->innerMomentum().z() );
	    outerRecTrack.fV = TEveVector( muon->track()->innerPosition().x(),
                                          muon->track()->innerPosition().y(),
                                          muon->track()->innerPosition().z() );
	    outerRecTrack.fSign = muon->charge();
	    outerTrackIsInitialized = true;
	 } else {
	    // bad case since single track is used.
	    outerRecTrack.fP = TEveVector( -muon->track()->px(),
                                          -muon->track()->py(),
                                          -muon->track()->pz() );
	    outerRecTrack.fV = TEveVector( muon->track()->vertex().x(),
                                          muon->track()->vertex().y(),
                                          muon->track()->vertex().z() );
	    outerRecTrack.fSign = muon->charge();
	    outerTrackIsInitialized = true;
	 }
      } else {
	 // track points in the right direction
	 // change type of the last point to daughter
	 // and set last points
	 // change last pathmark type
	 if ( !trk->RefPathMarks().empty())
            trk->RefPathMarks().back().fType = TEvePathMark::kDaughter;
	 if ( useStandAloneFit ) {
	    TEvePathMark mark( TEvePathMark::kDecay );
	    if (  muon->standAloneMuon()->innerPosition().R() >
                muon->standAloneMuon()->outerPosition().R() ) {
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
	    trk->AddPathMark( mark );
	 } else {
	    // muon match info
	    TEvePathMark mark( TEvePathMark::kDecay );
	    mark.fV = firstMatch( &*muon, iItem );
	    trk->AddPathMark( mark );
	 }
	 trk->SetPropagator( m_innerPropagator.get() );
	 trk->MakeTrack();
	 muonList->AddElement( trk );
	 // get last two points of the innerTrack trajectory
	 trk->GetPoint( trk->GetLastPoint(),   lastPointVX2, lastPointVY2, lastPointVZ2);
	 trk->GetPoint( trk->GetLastPoint()-1, lastPointVX1, lastPointVY1, lastPointVZ1);
	 useLastPoint = true;
      }
   }
   
   if ( useLastPoint ) {
      outerRecTrack.fV = TEveVector(lastPointVX2,lastPointVY2,lastPointVZ2);
      float scale = muon->p4().P()/sqrt( (lastPointVX2-lastPointVX1)*(lastPointVX2-lastPointVX1) + (lastPointVY2-lastPointVY1)*(lastPointVY2-lastPointVY1) + (lastPointVZ2-lastPointVZ1)*(lastPointVZ2-lastPointVZ1) );
      outerRecTrack.fP = TEveVector(scale*(lastPointVX2-lastPointVX1), scale*(lastPointVY2-lastPointVY1),scale*(lastPointVZ2-lastPointVZ1));
      outerTrackIsInitialized = true;
   }
   
   if ( muon->isTrackerMuon() && ! useStandAloneFit ) {
      outerRecTrack.fSign = innerRecTrack.fSign;
      TEveTrack* outerTrack = new TEveTrack( &outerRecTrack, m_outerPropagator.get() );
      outerTrack->SetMainColor( iItem->defaultDisplayProperties().color() );
      // std::cout << "\tpx " << outerRecTrack.fP.fX << " py " << outerRecTrack.fP.fY << " pz " << outerRecTrack.fP.fZ
      //  << " lastPointVX " << outerRecTrack.fV.fX << " vy " << outerRecTrack.fV.fY << " vz " << outerRecTrack.fV.fZ
      //  << " sign " << outerRecTrack.fSign << std::endl;
      //
      // add muon segments
      addMatchInformation( &(*muon), iItem, outerTrack, muonList, showEndcap );
      // change last pathmark type
      if ( !outerTrack->RefPathMarks().empty())
         outerTrack->RefPathMarks().back().fType = TEvePathMark::kDecay;
      outerTrack->MakeTrack();
      muonList->AddElement( outerTrack );
   }
   
   if ( useStandAloneFit )
   {
      if ( ! outerTrackIsInitialized ) {
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
      
      TEveTrack* outerTrack = new TEveTrack( &outerRecTrack, m_outerPropagator.get() );
      outerTrack->SetMainColor( iItem->defaultDisplayProperties().color() );
      
      TEvePathMark mark1( TEvePathMark::kDaughter );
      TEvePathMark mark2( TEvePathMark::kDecay );
      if (  muon->standAloneMuon()->innerPosition().R() <  muon->standAloneMuon()->outerPosition().R() ) {
         mark1.fV = TEveVector( muon->standAloneMuon()->innerPosition().x(),
                               muon->standAloneMuon()->innerPosition().y(),
                               muon->standAloneMuon()->innerPosition().z() );
         mark1.fTime = muon->standAloneMuon()->innerPosition().R();
         mark2.fV = TEveVector( muon->standAloneMuon()->outerPosition().x(),
                               muon->standAloneMuon()->outerPosition().y(),
                               muon->standAloneMuon()->outerPosition().z() );
         mark2.fTime = muon->standAloneMuon()->outerPosition().R();
      } else {
         mark1.fV = TEveVector( muon->standAloneMuon()->outerPosition().x(),
                               muon->standAloneMuon()->outerPosition().y(),
                               muon->standAloneMuon()->outerPosition().z() );
         mark1.fTime = muon->standAloneMuon()->outerPosition().R();
         mark2.fV = TEveVector( muon->standAloneMuon()->innerPosition().x(),
                               muon->standAloneMuon()->innerPosition().y(),
                               muon->standAloneMuon()->innerPosition().z() );
         mark2.fTime = muon->standAloneMuon()->innerPosition().R();
      }
      if ( mark1.fV.Perp() > outerTrack->GetVertex().Perp() ) outerTrack->AddPathMark( mark1 );
      outerTrack->AddPathMark( mark2 );
      
      outerTrack->MakeTrack();
      muonList->AddElement( outerTrack );
   }
}

//
// const member functions
//

//
// static member functions
//
