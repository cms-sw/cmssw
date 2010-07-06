// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonBuilder
// $Id: FWMuonBuilder.cc,v 1.28 2010/06/18 12:44:06 yana Exp $
//

#include "TEveVSDStructs.h"
#include "TEveTrack.h"
#include "TEveStraightLineSet.h"
#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"

#include "Fireworks/Muons/interface/FWMuonBuilder.h"
#include "Fireworks/Muons/interface/SegmentUtils.h"
#include "Fireworks/Muons/interface/CSCUtils.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

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
                          FWProxyBuilderBase* pb,
                          TEveElement* parentList,
                          bool showEndcap)
{
  std::set<unsigned int> ids;
  const DetIdToMatrix* geom = pb->context().getGeom();
  
  const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
   
  //need to use auto_ptr since the segmentSet may not be passed to muonList
  std::auto_ptr<TEveStraightLineSet> segmentSet(new TEveStraightLineSet);
  segmentSet->SetLineWidth(4);

  for ( std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin(), 
                                                        chambersEnd = matches.end(); 
        chamber != chambersEnd; ++chamber )
  {
    DetId id = chamber->id;

    if ( ids.insert(id.rawId()).second &&  // ensure that we add same chamber only once
         ( chamber->detector() != MuonSubdetId::CSC || showEndcap ) )
    {
      TEveGeoShape* shape = geom->getShape(id.rawId());
   
      if (shape) 
      {
        shape->SetElementName("Chamber");
        shape->RefMainTrans().Scale(0.999, 0.999, 0.999);
        pb->setupAddElement(shape, parentList);
      }

      else
      {
        fwLog(fwlog::kWarning) 
          <<"ERROR: failed to get shape of muon chamber with detid: "
          << id.rawId() <<std::endl;
      }
    }
     
    const TGeoHMatrix* matrix = geom->getMatrix(id.rawId());
    
    if ( ! matrix )
    {
      fwLog(fwlog::kError) <<" failed to get matrix for muon chamber with detid: "
                           << id.rawId() <<std::endl;
      return;
    }

    for( std::vector<reco::MuonSegmentMatch>::const_iterator segment = chamber->segmentMatches.begin(),
                                                          segmentEnd = chamber->segmentMatches.end();
         segment != segmentEnd; ++segment )
    {

      double segmentLength = 0.0;
      double segmentLimit  = 0.0;

      if ( chamber->detector() == MuonSubdetId::DT )
        segmentLength = 17.0; // FIXME: Can we get this from TGeoShape?

      else if ( chamber->detector() == MuonSubdetId::CSC )
      {
        CSCDetId cscDetId(id);

        double length    = 0.0;
        double thickness = 0.0;
        
        // FIXME: Can we get this information from TGeoShape?
        fireworks::fillCSCChamberParameters(cscDetId.station(), 
                                            cscDetId.ring(), 
                                            length, thickness);

        segmentLength = thickness*0.5;
        segmentLimit  = length*0.5;

        // Check if CSC segment position lies outside the chamber: a pathology of the reconstruction.
        // If so, do not draw segment.

        if ( fabs(segment->y) > segmentLimit )
        {
          fwLog(fwlog::kWarning) <<" position of CSC segment lies outside the chamber; station: "
                                 << cscDetId.station() <<"  ring: "<< cscDetId.ring() << std::endl;   
          
          continue;
        }
      }
      
      else
      {
        fwLog(fwlog::kWarning) <<" MuonSubdetId: "<< chamber->detector() <<std::endl;
        continue;
      }

      double segmentPosition[3] = 
      {
        segment->x, segment->y, 0.0
      };
      
      double segmentDirection[3] = 
      {
        segment->dXdZ, segment->dYdZ, 0.0
      };

      double localSegmentInnerPoint[3];
      double localSegmentOuterPoint[3];

      fireworks::createSegment(chamber->detector(), true, 
                               segmentLength, segmentLimit, 
                               segmentPosition, segmentDirection,
                               localSegmentInnerPoint, localSegmentOuterPoint);
                               
      double localSegmentCenterPoint[3] = 
      {
        segment->x, segment->y, 0.0
      };

      double globalSegmentInnerPoint[3];
      double globalSegmentCenterPoint[3];
      double globalSegmentOuterPoint[3];
      
      matrix->LocalToMaster( localSegmentInnerPoint, globalSegmentInnerPoint );
      matrix->LocalToMaster( localSegmentCenterPoint, globalSegmentCenterPoint );
      matrix->LocalToMaster( localSegmentOuterPoint, globalSegmentOuterPoint );
         
      if( globalSegmentInnerPoint[1]*globalSegmentOuterPoint[1] > 0 ) 
      {
        segmentSet->AddLine( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
                             globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
      }         

      else 
      {
        if( fabs(globalSegmentInnerPoint[1]) > fabs(globalSegmentOuterPoint[1]) )
          segmentSet->AddLine( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
                               globalSegmentCenterPoint[0], globalSegmentCenterPoint[1], globalSegmentCenterPoint[2] );
        else
          segmentSet->AddLine( globalSegmentCenterPoint[0], globalSegmentCenterPoint[1], globalSegmentCenterPoint[2],
                               globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
      }               
    } 
  }
   
  if ( !matches.empty() ) 
    pb->setupAddElement( segmentSet.release(), parentList );
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
}

FWMuonBuilder::~FWMuonBuilder()
{
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
   if ( field->getSource() == FWMagField::kNone ) {
      if ( fabs( iData.eta() ) > 2.0 || iData.pt() < 3 ) return;
      if ( iData.innerTrack().isAvailable() ){
         double estimate = fw::estimate_field( *( iData.innerTrack() ), true );
         if ( estimate >= 0 ) field->guessField( estimate );
	 
      }
      if ( iData.outerTrack().isAvailable() ){
         double estimate = fw::estimate_field( *( iData.outerTrack() ) );
         if ( estimate >= 0 ) field->guessFieldIsOn( estimate > 0.5 );
      }
   }
}

//______________________________________________________________________________

void
FWMuonBuilder::buildMuon(FWProxyBuilderBase* pb,
                         const reco::Muon* muon,
                         TEveElement* tList,
                         bool showEndcap,
                         bool tracksOnly)
{
   calculateField(*muon, pb->context().getField());

  
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
	!buggyMuon( &*muon, pb->context().getGeom() ) )
   {
      TEveTrack* trk = fireworks::prepareTrack(*(muon->innerTrack()),
                                               pb->context().getMuonTrackPropagator(),
					       getRecoTrajectoryPoints(muon,pb->item()) );
      trk->MakeTrack();
      pb->setupAddElement(trk, tList);
      if ( ! tracksOnly )
	 addMatchInformation( &(*muon), pb, tList, showEndcap );
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
                                               pb->context().getMuonTrackPropagator(),
                                               extraPoints);
      trk->MakeTrack();
      pb->setupAddElement(trk, tList);
      return;
   }

   if ( muon->innerTrack().isAvailable() )
   {
      TEveTrack* trk = fireworks::prepareTrack(*(muon->innerTrack()), pb->context().getMuonTrackPropagator());
      trk->MakeTrack();
      pb->setupAddElement(trk, tList);
      return;
   }

   if ( muon->outerTrack().isAvailable() )
   {
      TEveTrack* trk = fireworks::prepareTrack(*(muon->outerTrack()), pb->context().getMuonTrackPropagator());
      trk->MakeTrack();
      pb->setupAddElement(trk, tList);
      return;
   }
   
   // if got that far it means we have nothing but a candidate
   // show it anyway.
   TEveTrack* trk = fireworks::prepareCandidate(*muon, pb->context().getMuonTrackPropagator());
   trk->MakeTrack();
   pb->setupAddElement(trk, tList);
}
