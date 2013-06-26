/*
 *  FWPSimHitProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/9/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "TEvePointSet.h"

class FWPSimHitProxyBuilder : public FWSimpleProxyBuilderTemplate<PSimHit>
{
public:
   FWPSimHitProxyBuilder( void ) {} 
   virtual ~FWPSimHitProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWPSimHitProxyBuilder( const FWPSimHitProxyBuilder& );
   // Disable default assignment operator
   const FWPSimHitProxyBuilder& operator=( const FWPSimHitProxyBuilder& );

   void build( const PSimHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWPSimHitProxyBuilder::build( const PSimHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
   TEvePointSet* pointSet = new TEvePointSet;
   setupAddElement( pointSet, &oItemHolder );
   const FWGeometry *geom = item()->getGeom();
   unsigned int rawid = iData.detUnitId();
   if( ! geom->contains( rawid ))
   {
      fwLog( fwlog::kError )
	<< "failed to get geometry of detid: " 
	<< rawid << std::endl;
      return;
   }
   
   float local[3] = { iData.localPosition().x(), iData.localPosition().y(), iData.localPosition().z() };
   float global[3];
   geom->localToGlobal( rawid, local, global );
   pointSet->SetNextPoint( global[0], global[1], global[2] );
}

REGISTER_FWPROXYBUILDER( FWPSimHitProxyBuilder, PSimHit, "PSimHits", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
