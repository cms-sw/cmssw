/*
 *  FWEcalRecHitEtProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 5/28/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

class FWEcalRecHitEtProxyBuilder : public FWSimpleProxyBuilderTemplate<EcalRecHit>
{
public:
   FWEcalRecHitEtProxyBuilder( void ) {}
	
   virtual ~FWEcalRecHitEtProxyBuilder( void ) {}
	
   REGISTER_PROXYBUILDER_METHODS();
	
private:
   // Disable default copy constructor
   FWEcalRecHitEtProxyBuilder( const FWEcalRecHitEtProxyBuilder& );
   // Disable default assignment operator
   const FWEcalRecHitEtProxyBuilder& operator=( const FWEcalRecHitEtProxyBuilder& );
	
   void build( const EcalRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWEcalRecHitEtProxyBuilder::build( const EcalRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
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
   
   fireworks::drawEtTower3D( corners, iData.energy() * scale, &oItemHolder, this, reflect );
}

REGISTER_FWPROXYBUILDER( FWEcalRecHitEtProxyBuilder, EcalRecHit, "Ecal RecHit Et", FWViewType::kISpyBit );
