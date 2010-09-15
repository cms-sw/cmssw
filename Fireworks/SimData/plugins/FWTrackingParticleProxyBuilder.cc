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
#include "TParticle.h"

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
   TParticle* particle = new TParticle;
   particle->SetPdgCode( iData.pdgId());
   particle->SetMomentum( iData.px(), iData.py(), iData.pz(), iData.p());
   // particle->SetProductionVertex( iData.vx(), iData.vy(), iData.vz(), 0.0 );
   particle->SetProductionVertex( iData.parentVertex()->position().x() / 100.0,
				  iData.parentVertex()->position().y() / 100.0,
				  iData.parentVertex()->position().z() / 100.0, 0.0 );
   TEveTrackPropagator* propagator = context().getTrackPropagator();
  
   TEveTrack* track = new TEveTrack( particle, iIndex, propagator );
   setupAddElement( track, &oItemHolder );

   TEvePointSet* pointSet = new TEvePointSet;
   setupAddElement( pointSet, &oItemHolder );
   const FWGeometry *geom = item()->getGeom();
   const std::vector<PSimHit>& hits = iData.trackPSimHit();
   for( std::vector<PSimHit>::const_iterator it = hits.begin(), end = hits.end(); it != end; ++it )
   {
     const PSimHit& phit = (*it);
     float local[3] = { phit.localPosition().x(), phit.localPosition().y(), phit.localPosition().z() };
     float global[3];
     geom->localToGlobal( phit.detUnitId(), local, global );
     pointSet->SetNextPoint( global[0], global[1], global[2] );
   }
}

REGISTER_FWPROXYBUILDER( FWTrackingParticleProxyBuilder, TrackingParticle, "TrackingParticles", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
