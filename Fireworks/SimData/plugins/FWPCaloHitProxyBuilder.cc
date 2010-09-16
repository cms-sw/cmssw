/*
 *  FWPCaloHitProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/9/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

class FWPCaloHitProxyBuilder : public FWSimpleProxyBuilderTemplate<PCaloHit>
{
public:
   FWPCaloHitProxyBuilder( void ) {} 
   virtual ~FWPCaloHitProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWPCaloHitProxyBuilder( const FWPCaloHitProxyBuilder& );
   // Disable default assignment operator
   const FWPCaloHitProxyBuilder& operator=( const FWPCaloHitProxyBuilder& );

   void build( const PCaloHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWPCaloHitProxyBuilder::build( const PCaloHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
   const float* corners = item()->getGeom()->getCorners( iData.id());
   if( corners == 0 ) {
      return;
   }
   Float_t scale = 10.0;
   fireworks::drawEnergyTower3D( corners, iData.energy() * scale, &oItemHolder, this, false );
}

REGISTER_FWPROXYBUILDER( FWPCaloHitProxyBuilder, PCaloHit, "PCaloHits", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
