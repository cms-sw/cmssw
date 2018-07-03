/*
 *  FWTrackingVertexProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 10/6/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include "TEveTrack.h"

class FWTrackingVertexProxyBuilder : public FWSimpleProxyBuilderTemplate<TrackingVertex>
{
public:
   FWTrackingVertexProxyBuilder( void ) {} 
   ~FWTrackingVertexProxyBuilder( void ) override {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWTrackingVertexProxyBuilder( const FWTrackingVertexProxyBuilder& ) = delete;
   // Disable default assignment operator
   const FWTrackingVertexProxyBuilder& operator=( const FWTrackingVertexProxyBuilder& ) = delete;

   using FWSimpleProxyBuilderTemplate<TrackingVertex>::build;
   void build( const TrackingVertex& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;
};

void
FWTrackingVertexProxyBuilder::build( const TrackingVertex& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
   TEvePointSet* pointSet = new TEvePointSet;
   setupAddElement( pointSet, &oItemHolder );
   pointSet->SetNextPoint( iData.position().x(), iData.position().y(), iData.position().z() );
}

REGISTER_FWPROXYBUILDER( FWTrackingVertexProxyBuilder, TrackingVertex, "TrackingVertices", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
