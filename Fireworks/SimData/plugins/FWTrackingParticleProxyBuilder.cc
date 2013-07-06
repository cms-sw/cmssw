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
   
   TEvePointSet* pointSet = new TEvePointSet;
   setupAddElement( pointSet, track );
   pointSet->SetMarkerSize(item()->getConfig()->value<long>("Point Size"));
#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
   const FWGeometry *geom = item()->getGeom();
   const std::vector<PSimHit>& hits = iData.trackPSimHit();

   float local[3];
   float localDir[3];
   float global[3] = { 0.0, 0.0, 0.0 };
   float globalDir[3] = { 0.0, 0.0, 0.0 };
   std::vector<PSimHit>::const_iterator it = hits.begin();
   std::vector<PSimHit>::const_iterator end = hits.end();
   if( it != end )
   {
      unsigned int trackid = hits.begin()->trackId();

      for( ; it != end; ++it )
      {
	 const PSimHit& phit = (*it);
	 if( phit.trackId() != trackid )
	 {
	    trackid = phit.trackId();
	    track->AddPathMark( TEvePathMark( TEvePathMark::kDecay, TEveVector( global[0], global[1], global[2] ),
					      TEveVector( globalDir[0], globalDir[1], globalDir[2] )));
	 }
	 local[0] = phit.localPosition().x();
	 local[1] = phit.localPosition().y();
	 local[2] = phit.localPosition().z();
	 localDir[0] = phit.momentumAtEntry().x();
	 localDir[1] = phit.momentumAtEntry().y();
	 localDir[2] = phit.momentumAtEntry().z();
	 geom->localToGlobal( phit.detUnitId(), local, global );
	 geom->localToGlobal( phit.detUnitId(), localDir, globalDir );
	 pointSet->SetNextPoint( global[0], global[1], global[2] );
	 track->AddPathMark( TEvePathMark( TEvePathMark::kReference/*kDaughter*/, TEveVector( global[0], global[1], global[2] ),
					   TEveVector( globalDir[0], globalDir[1], globalDir[2] )));
      }
      if( hits.size() > 1 )
	 track->AddPathMark( TEvePathMark( TEvePathMark::kDecay, TEveVector( global[0], global[1], global[2] ),
					   TEveVector( globalDir[0], globalDir[1], globalDir[2] )));
   }
#endif
   
   track->MakeTrack();
   setupAddElement( track, &oItemHolder );
}

REGISTER_FWPROXYBUILDER( FWTrackingParticleProxyBuilder, TrackingParticle, "TrackingParticles", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
