/*
 *  FWCastorRecHitProxyBuilder.cc
 *  cmsShow
 *
 *  Created by Ianna Osborne on 7/8/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

class FWCastorRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<CastorRecHit>
{
public:
   FWCastorRecHitProxyBuilder( void ) {}  
   virtual ~FWCastorRecHitProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWCastorRecHitProxyBuilder( const FWCastorRecHitProxyBuilder& );
   // Disable default assignment operator
   const FWCastorRecHitProxyBuilder& operator=( const FWCastorRecHitProxyBuilder& );

   void build( const CastorRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWCastorRecHitProxyBuilder::build( const CastorRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
   std::vector<TEveVector> corners = item()->getGeom()->getPoints( iData.detid());
   if( corners.empty() ) {
      return;
   }
   // FIXME: Every other shapes is inverted.
   fireworks::drawEnergyTower3D( corners, iData.energy() * 10, &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWCastorRecHitProxyBuilder, CastorRecHit, "Castor RecHit", FWViewType::kISpyBit );
