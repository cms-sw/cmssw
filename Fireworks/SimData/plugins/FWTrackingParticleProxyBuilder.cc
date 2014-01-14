/*
 *  FWTrackingParticleProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/9/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWParameters.h"

#include "TEveTrack.h"

class FWTrackingParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<TrackingParticle>
{
public:
   FWTrackingParticleProxyBuilder( void ) {} 
   virtual ~FWTrackingParticleProxyBuilder( void ) {}

   virtual void setItem(const FWEventItem* iItem) {
      FWProxyBuilderBase::setItem(iItem);
      iItem->getConfig()->assertParam("Point Size", 1l, 3l, 1l);
   }

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
   t.fV = TEveVector( iData.vx(), iData.vy(), iData.vz() );
   t.fSign = iData.charge();
  
   TEveTrack* track = new TEveTrack(&t, context().getTrackPropagator());
   if( t.fSign == 0 )
      track->SetLineStyle( 7 );
   
   track->MakeTrack();
   setupAddElement( track, &oItemHolder );
}

REGISTER_FWPROXYBUILDER( FWTrackingParticleProxyBuilder, TrackingParticle, "TrackingParticles", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
