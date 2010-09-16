/*
 *  FWTrackingParticleProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/9/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "TEveTrack.h"

class FWTrackingParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<TrackingParticle>
{
public:
   FWTrackingParticleProxyBuilder( void ) {} 
   virtual ~FWTrackingParticleProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWTrackingParticleProxyBuilder( const FWTrackingParticleProxyBuilder& );
   // Disable default assignment operator
   const FWTrackingParticleProxyBuilder& operator=( const FWTrackingParticleProxyBuilder& );

   void build( const TrackingParticle& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWTrackingParticleProxyBuilder::build( const TrackingParticle& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
   TEveRecTrack t;
   t.fBeta = 1.0;
   t.fP = TEveVector( iData.px(), iData.py(), iData.pz() );
   t.fV = TEveVector( iData.vx() * 0.01, iData.vy() * 0.01, iData.vz() * 0.01 );
   t.fSign = iData.charge();
  
   TEveTrack* track = new TEveTrack(&t, context().getTrackPropagator());
   if( t.fSign == 0 )
      track->SetLineStyle( 7 );
   
   TEvePointSet* pointSet = new TEvePointSet;
   setupAddElement( pointSet, track );
   const FWGeometry *geom = item()->getGeom();
   const std::vector<PSimHit>& hits = iData.trackPSimHit();

   float local[3];
   float global[3];
   for( std::vector<PSimHit>::const_iterator it = hits.begin(), end = hits.end(); it != end; ++it )
   {
     const PSimHit& phit = (*it);
     local[0] = phit.localPosition().x();
     local[1] = phit.localPosition().y();
     local[2] = phit.localPosition().z();
     geom->localToGlobal( phit.detUnitId(), local, global );
     pointSet->SetNextPoint( global[0], global[1], global[2] );
     track->AddPathMark( TEvePathMark( TEvePathMark::kDaughter, TEveVector( global[0], global[1], global[2] )));
   }
   track->MakeTrack();
   setupAddElement( track, &oItemHolder );
}

REGISTER_FWPROXYBUILDER( FWTrackingParticleProxyBuilder, TrackingParticle, "TrackingParticles", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
