/*
 *  FWEcalRecHitProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 5/28/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

class FWEcalRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<EcalRecHit>
{
public:
   FWEcalRecHitProxyBuilder( void ) {}
	
   virtual ~FWEcalRecHitProxyBuilder( void ) {}
	
   REGISTER_PROXYBUILDER_METHODS();
	
private:
   // Disable default copy constructor
   FWEcalRecHitProxyBuilder( const FWEcalRecHitProxyBuilder& );
   // Disable default assignment operator
   const FWEcalRecHitProxyBuilder& operator=( const FWEcalRecHitProxyBuilder& );
	
   void build( const EcalRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWEcalRecHitProxyBuilder::build( const EcalRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
   const float* corners = item()->getGeom()->getCorners( iData.detid());
   if( corners == 0 ) {
      return;
   }
   Float_t scale = 10.0;
   bool reflect = false;
   if( EcalSubdetector( iData.detid().subdetId() ) == EcalPreshower )
   {
      scale = 1000.0; 	// FIXME: The scale should be taken form somewhere else
      ( corners[2] < 0.0 ) ? reflect = true : reflect = false;
   }
   
   fireworks::drawEnergyTower3D( corners, iData.energy() * scale, &oItemHolder, this, reflect );
}

REGISTER_FWPROXYBUILDER( FWEcalRecHitProxyBuilder, EcalRecHit, "Ecal RecHit", FWViewType::kISpyBit );
