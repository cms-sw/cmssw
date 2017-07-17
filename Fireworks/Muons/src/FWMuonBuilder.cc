// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonBuilder
//

#include "TEveVSDStructs.h"
#include "TEveTrack.h"
#include "TEveStraightLineSet.h"
#include "TEveGeoNode.h"
#include "TGeoArb8.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"

#include "Fireworks/Muons/interface/FWMuonBuilder.h"
#include "Fireworks/Muons/interface/SegmentUtils.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

namespace  {
std::vector<TEveVector> getRecoTrajectoryPoints( const reco::Muon* muon,
                                                 const FWEventItem* iItem )
{
   std::vector<TEveVector> points;
   const FWGeometry* geom = iItem->getGeom();
   
   float localTrajectoryPoint[3];
   float globalTrajectoryPoint[3];
   
   const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
   for( std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin(),
							 chamberEnd = matches.end();
	chamber != chamberEnd; ++chamber )
   {
      // expected track position
      localTrajectoryPoint[0] = chamber->x;
      localTrajectoryPoint[1] = chamber->y;
      localTrajectoryPoint[2] = 0;

      unsigned int rawid = chamber->id.rawId();
      if( geom->contains( rawid ))
      {
	 geom->localToGlobal( rawid, localTrajectoryPoint, globalTrajectoryPoint );
	 points.push_back( TEveVector(globalTrajectoryPoint[0],
				      globalTrajectoryPoint[1],
				      globalTrajectoryPoint[2] ));
      }
   }
   return points;
}

//______________________________________________________________________________

void addMatchInformation( const reco::Muon* muon,
                          FWProxyBuilderBase* pb,
                          TEveElement* parentList,
                          bool showEndcap )
{
  std::set<unsigned int> ids;
  const FWGeometry* geom = pb->context().getGeom();
  
  const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
   
  //need to use auto_ptr since the segmentSet may not be passed to muonList
  std::auto_ptr<TEveStraightLineSet> segmentSet( new TEveStraightLineSet );
  // FIXME: This should be set elsewhere.
  segmentSet->SetLineWidth( 4 );

  for( std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin(), 
						       chambersEnd = matches.end(); 
       chamber != chambersEnd; ++chamber )
  {
    unsigned int rawid = chamber->id.rawId();
    float segmentLength = 0.0;
    float segmentLimit  = 0.0;

    if( geom->contains( rawid ))
    {
      TEveGeoShape* shape = geom->getEveShape( rawid );
      shape->SetElementName( "Chamber" );
      shape->RefMainTrans().Scale( 0.999, 0.999, 0.999 );

      FWGeometry::IdToInfoItr det = geom->find( rawid );
      if( det->shape[0] == 1 ) // TGeoTrap
      {
        segmentLength = det->shape[3];
        segmentLimit  = det->shape[4];
      }
      else if( det->shape[0] == 2 ) // TGeoBBox
      {
	segmentLength = det->shape[3];
      }
      else
      {   
        const double segmentLength = 15;
        fwLog( fwlog::kWarning ) << Form("FWMuonBuilder: unknown shape type in muon chamber with detId=%d. Setting segment length to %.0f cm.\n",  rawid, segmentLength);
      }
        
      if( ids.insert( rawid ).second &&  // ensure that we add same chamber only once
	  ( chamber->detector() != MuonSubdetId::CSC || showEndcap ))
      {     
	pb->setupAddElement( shape, parentList );
      }
     
      for( std::vector<reco::MuonSegmentMatch>::const_iterator segment = chamber->segmentMatches.begin(),
							    segmentEnd = chamber->segmentMatches.end();
	   segment != segmentEnd; ++segment )
      {
	float segmentPosition[3]  = {    segment->x,     segment->y, 0.0 };
	float segmentDirection[3] = { segment->dXdZ,  segment->dYdZ, 0.0 };

	float localSegmentInnerPoint[3];
	float localSegmentOuterPoint[3];

	fireworks::createSegment( chamber->detector(), true, 
				  segmentLength, segmentLimit, 
				  segmentPosition, segmentDirection,
				  localSegmentInnerPoint, localSegmentOuterPoint );
      
	float globalSegmentInnerPoint[3];
	float globalSegmentOuterPoint[3];

	geom->localToGlobal( *det, localSegmentInnerPoint,  globalSegmentInnerPoint );
	geom->localToGlobal( *det, localSegmentOuterPoint,  globalSegmentOuterPoint );

	segmentSet->AddLine( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
			     globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
      }
    }
  }
  
  if( !matches.empty() ) 
    pb->setupAddElement( segmentSet.release(), parentList );
}

//______________________________________________________________________________

bool
buggyMuon( const reco::Muon* muon,
           const FWGeometry* geom )
{
   if( !muon->standAloneMuon().isAvailable() ||
       !muon->standAloneMuon()->extra().isAvailable())
     return false;
   
   float localTrajectoryPoint[3];
   float globalTrajectoryPoint[3];
   
   const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
   for( std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin(),
							 chamberEnd = matches.end();
	chamber != chamberEnd; ++chamber )
   {
      // expected track position
      localTrajectoryPoint[0] = chamber->x;
      localTrajectoryPoint[1] = chamber->y;
      localTrajectoryPoint[2] = 0;

      unsigned int rawid = chamber->id.rawId();
      if( geom->contains( rawid ))
      {
	 geom->localToGlobal( rawid, localTrajectoryPoint, globalTrajectoryPoint );
         double phi = atan2( globalTrajectoryPoint[1], globalTrajectoryPoint[0] );
         if( cos( phi - muon->standAloneMuon()->innerPosition().phi()) < 0 )
            return true;
      }
   }
   return false;
}

TEveTrack* prepareMuonTrackWithExtraPoints(const reco::Track& track,
              TEveTrackPropagator* propagator,
              const std::vector<TEveVector>& extraPoints )
{
   TEveRecTrack t;
   t.fBeta = 1.;
   t.fSign = track.charge();
   t.fV.Set(track.vx(), track.vy(), track.vz());
   t.fP.Set(track.px(), track.py(), track.pz());
   //  t.fSign = muon->charge();
   //  t.fV.Set(muon->vx(), muon->vy(), muon->vz());
   //  t.fP.Set(muon->px(), muon->py(), muon->pz());
   TEveTrack* trk = new TEveTrack( &t, propagator );
   size_t n = extraPoints.size();

   if (n > 1) {
      int lastDaughter = n-2;
      for (int i = 0; i <= lastDaughter; ++i)
         trk->AddPathMark( TEvePathMark( TEvePathMark::kDaughter, extraPoints[i] ) );
   }
   trk->AddPathMark( TEvePathMark( TEvePathMark::kDecay, extraPoints.back() ) );
   return trk;
}


}

//
// constructors and destructor
//
FWMuonBuilder::FWMuonBuilder():m_lineWidth(1)
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
FWMuonBuilder::calculateField( const reco::Muon& iData, FWMagField* field )
{
   // if auto field estimation mode, do extra loop over muons.
   // use both inner and outer track if available
   if( field->getSource() == FWMagField::kNone )
   {
      if( fabs( iData.eta() ) > 2.0 || iData.pt() < 3 ) return;
      if( iData.innerTrack().isAvailable())
      {
         double estimate = fw::estimate_field( *( iData.innerTrack()), true );
         if( estimate >= 0 ) field->guessField( estimate );	 
      }
      if( iData.outerTrack().isAvailable() )
      {
         double estimate = fw::estimate_field( *( iData.outerTrack()));
         if( estimate >= 0 ) field->guessFieldIsOn( estimate > 0.5 );
      }
   }
}

//______________________________________________________________________________

void
FWMuonBuilder::buildMuon( FWProxyBuilderBase* pb,
			  const reco::Muon* muon,
			  TEveElement* tList,
			  bool showEndcap,
			  bool tracksOnly )
{
   calculateField( *muon, pb->context().getField());
  
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

   if( muon->isTrackerMuon() && 
       muon->innerTrack().isAvailable() &&
       muon->isMatchesValid() &&
       !buggyMuon( &*muon, pb->context().getGeom()))
   {
      TEveTrack* trk = fireworks::prepareTrack( *(muon->innerTrack()),
						pb->context().getMuonTrackPropagator(),
						getRecoTrajectoryPoints( muon, pb->item()));
      trk->MakeTrack();
      trk->SetLineWidth(m_lineWidth);
      pb->setupAddElement( trk, tList );
      if( ! tracksOnly )
	 addMatchInformation( &(*muon), pb, tList, showEndcap );
      return;
   } 

   if( muon->isGlobalMuon() &&
       muon->globalTrack().isAvailable())
   {
      std::vector<TEveVector> extraPoints;
      if( muon->innerTrack().isAvailable() &&  muon->innerTrack()->extra().isAvailable())
      {
	 extraPoints.push_back( TEveVector( muon->innerTrack()->innerPosition().x(),
					    muon->innerTrack()->innerPosition().y(),
					    muon->innerTrack()->innerPosition().z()));
	 extraPoints.push_back( TEveVector( muon->innerTrack()->outerPosition().x(),
					    muon->innerTrack()->outerPosition().y(),
					    muon->innerTrack()->outerPosition().z()));
      }
      if( muon->outerTrack().isAvailable() &&  muon->outerTrack()->extra().isAvailable())
      {
	 extraPoints.push_back( TEveVector( muon->outerTrack()->innerPosition().x(),
					    muon->outerTrack()->innerPosition().y(),
					    muon->outerTrack()->innerPosition().z()));
	 extraPoints.push_back( TEveVector( muon->outerTrack()->outerPosition().x(),
					    muon->outerTrack()->outerPosition().y(),
					    muon->outerTrack()->outerPosition().z()));
      }
      TEveTrack* trk = 0;
      if (extraPoints.empty())
         trk = fireworks::prepareTrack( *( muon->globalTrack()),pb->context().getMuonTrackPropagator());
      else
         trk = prepareMuonTrackWithExtraPoints(*( muon->globalTrack()),pb->context().getMuonTrackPropagator(), extraPoints); 

      trk->MakeTrack();
      trk->SetLineWidth(m_lineWidth);
      pb->setupAddElement( trk, tList );
      return;
   }

   if( muon->innerTrack().isAvailable())
   {
      TEveTrack* trk = fireworks::prepareTrack( *( muon->innerTrack()), pb->context().getMuonTrackPropagator());
      trk->MakeTrack();
      pb->setupAddElement( trk, tList );
      return;
   }

   if( muon->outerTrack().isAvailable())
   {
      TEveTrack* trk = fireworks::prepareTrack( *( muon->outerTrack()), pb->context().getMuonTrackPropagator());
      trk->MakeTrack();
      trk->SetLineWidth(m_lineWidth);
      pb->setupAddElement( trk, tList );
      return;
   }
   
   // if got that far it means we have nothing but a candidate
   // show it anyway.
   TEveTrack* trk = fireworks::prepareCandidate( *muon, pb->context().getMuonTrackPropagator());
   trk->MakeTrack();
   trk->SetLineWidth(m_lineWidth);
   pb->setupAddElement( trk, tList );
}
