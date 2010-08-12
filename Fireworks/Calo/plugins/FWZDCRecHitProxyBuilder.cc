/*
 *  FWZDCRecHitProxyBuilder.cc
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
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"

class FWZDCRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<ZDCRecHit>
{
public:
   FWZDCRecHitProxyBuilder( void ) {}  
   virtual ~FWZDCRecHitProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWZDCRecHitProxyBuilder( const FWZDCRecHitProxyBuilder& );
   // Disable default assignment operator
   const FWZDCRecHitProxyBuilder& operator=( const FWZDCRecHitProxyBuilder& );

   void build( const ZDCRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWZDCRecHitProxyBuilder::build( const ZDCRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
   const std::vector<Float_t>& corners = item()->getGeom()->getCorners( iData.detid());
   if( corners.empty()) {
      return;
   }
  
   fireworks::drawEnergyTower3D( corners, iData.energy(), &oItemHolder, this );
}

REGISTER_FWPROXYBUILDER( FWZDCRecHitProxyBuilder, ZDCRecHit, "ZDC RecHit", FWViewType::kISpyBit );
