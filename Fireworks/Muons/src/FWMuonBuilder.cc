// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonBuilder
// $Id: FWMuonBuilder.cc,v 1.29 2010/07/06 18:32:08 amraktad Exp $
//

#include "TEveVSDStructs.h"
#include "TEveTrack.h"
#include "TEveStraightLineSet.h"
#include "TEveGeoNode.h"
#include "TGeoArb8.h"

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

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

namespace  {
std::vector<TEveVector> getRecoTrajectoryPoints( const reco::Muon* muon,
                                                 const FWEventItem* iItem )
{
   std::vector<TEveVector> points;
   const DetIdToMatrix* geom = iItem->getGeom();
   
   Double_t localTrajectoryPoint[3];
   Double_t globalTrajectoryPoint[3];
   
   const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
   for( std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin(),
							 chamberEnd = matches.end();
	chamber != matches.end(); ++chamber )
   {
      // expected track position
      localTrajectoryPoint[0] = chamber->x;
      localTrajectoryPoint[1] = chamber->y;
      localTrajectoryPoint[2] = 0;

      const TGeoHMatrix* matrix = geom->getMatrix( chamber->id.rawId());
      if ( matrix )
      {
         matrix->LocalToMaster( localTrajectoryPoint, globalTrajectoryPoint );
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
  const DetIdToMatrix* geom = pb->context().getGeom();
  
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
    double segmentLength = 0.0;
    double segmentLimit  = 0.0;

    TEveGeoShape* shape = geom->getShape( rawid );
    if( shape ) 
    {
      shape->SetElementName( "Chamber" );
      shape->RefMainTrans().Scale( 0.999, 0.999, 0.999 );

      if( TGeoTrap* trap = dynamic_cast<TGeoTrap*>( shape->GetShape()))
      {
	Double_t thickness = trap->GetDz();
	Double_t apothem = trap->GetH1();
	Double_t hBottomEdge = trap->GetBl1();
	Double_t hTopEdge = trap->GetTl1();

	fwLog( fwlog::kDebug )
	  << "thickness = " << thickness
	  << ", apothem = " << apothem
	  << ", hBottomEdge = " << hBottomEdge
	  << ", hTopEdge = " << hTopEdge << std::endl;

        segmentLength = thickness;
        segmentLimit  = apothem;
      }
      else if( TGeoBBox* box = dynamic_cast<TGeoBBox*>( shape->GetShape()))
      {
	Double_t width = box->GetDX();
	Double_t length = box->GetDY();
	Double_t thickness = box->GetDZ();
	segmentLength = thickness;
	
	fwLog( fwlog::kDebug )
	  << "width = " << width
	  << ", length = " << length
	  << ", thickness = " << thickness << std::endl;
      }
      else
      {	
	fwLog( fwlog::kWarning ) 
	  << "failed to get segment limits for detid: "
	  << rawid << std::endl;
      }
    }
    else
    {
      fwLog( fwlog::kWarning ) 
	<< "failed to get shape of muon chamber with detid: "
	<< rawid << std::endl;
    }
        
    if( ids.insert( rawid ).second &&  // ensure that we add same chamber only once
	( chamber->detector() != MuonSubdetId::CSC || showEndcap ))
    {     
      pb->setupAddElement( shape, parentList );
    }
     
    const TGeoHMatrix* matrix = geom->getMatrix( rawid );
    
    if( ! matrix )
    {
      fwLog( fwlog::kError )
	<< "failed to get matrix for muon chamber with detid: "
	<< rawid << std::endl;
      return;
    }

    for( std::vector<reco::MuonSegmentMatch>::const_iterator segment = chamber->segmentMatches.begin(),
                                                          segmentEnd = chamber->segmentMatches.end();
         segment != segmentEnd; ++segment )
    {
      double segmentPosition[3]  = {    segment->x,    segment->y, 0.0 };
      double segmentDirection[3] = { segment->dXdZ, segment->dYdZ, 0.0 };

      double localSegmentInnerPoint[3];
      double localSegmentOuterPoint[3];

      fireworks::createSegment( chamber->detector(), true, 
				segmentLength, segmentLimit, 
				segmentPosition, segmentDirection,
				localSegmentInnerPoint, localSegmentOuterPoint );
      
      fireworks::addSegment( matrix, *segmentSet, segmentPosition, localSegmentInnerPoint, localSegmentOuterPoint );
    } 
  }
   
  if( !matches.empty() ) 
    pb->setupAddElement( segmentSet.release(), parentList );
}

//______________________________________________________________________________

bool
buggyMuon( const reco::Muon* muon,
           const DetIdToMatrix* geom )
{
   if( !muon->standAloneMuon().isAvailable() ||
       !muon->standAloneMuon()->extra().isAvailable())
     return false;
   
   Double_t localTrajectoryPoint[3];
   Double_t globalTrajectoryPoint[3];
   
   const std::vector<reco::MuonChamberMatch>& matches = muon->matches();
   for( std::vector<reco::MuonChamberMatch>::const_iterator chamber = matches.begin(),
							 chamberEnd = matches.end();
	chamber != chamberEnd; ++chamber )
   {
      // expected track position
      localTrajectoryPoint[0] = chamber->x;
      localTrajectoryPoint[1] = chamber->y;
      localTrajectoryPoint[2] = 0;

      const TGeoHMatrix* matrix = geom->getMatrix( chamber->id.rawId());
      if( matrix )
      {
         matrix->LocalToMaster( localTrajectoryPoint, globalTrajectoryPoint );
         double phi = atan2( globalTrajectoryPoint[1], globalTrajectoryPoint[0] );
         if( cos( phi - muon->standAloneMuon()->innerPosition().phi()) < 0 )
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
      pb->setupAddElement( trk, tList );
      if( ! tracksOnly )
	 addMatchInformation( &(*muon), pb, tList, showEndcap );
      return;
   } 

   if( muon->isGlobalMuon() &&
       muon->globalTrack().isAvailable())
   {
      std::vector<TEveVector> extraPoints;
      if( muon->innerTrack().isAvailable())
      {
	 extraPoints.push_back( TEveVector( muon->innerTrack()->innerPosition().x(),
					    muon->innerTrack()->innerPosition().y(),
					    muon->innerTrack()->innerPosition().z()));
	 extraPoints.push_back( TEveVector( muon->innerTrack()->outerPosition().x(),
					    muon->innerTrack()->outerPosition().y(),
					    muon->innerTrack()->outerPosition().z()));
      }
      if( muon->outerTrack().isAvailable())
      {
	 extraPoints.push_back( TEveVector( muon->outerTrack()->innerPosition().x(),
					    muon->outerTrack()->innerPosition().y(),
					    muon->outerTrack()->innerPosition().z()));
	 extraPoints.push_back( TEveVector( muon->outerTrack()->outerPosition().x(),
					    muon->outerTrack()->outerPosition().y(),
					    muon->outerTrack()->outerPosition().z()));
      }
      TEveTrack* trk = fireworks::prepareTrack( *( muon->globalTrack()),
						pb->context().getMuonTrackPropagator(),
						extraPoints );
      trk->MakeTrack();
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
      pb->setupAddElement( trk, tList );
      return;
   }
   
   // if got that far it means we have nothing but a candidate
   // show it anyway.
   TEveTrack* trk = fireworks::prepareCandidate( *muon, pb->context().getMuonTrackPropagator());
   trk->MakeTrack();
   pb->setupAddElement( trk, tList );
}
