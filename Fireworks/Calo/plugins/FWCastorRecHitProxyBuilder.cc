/*
 *  FWCastorRecHitProxyBuilder.cc
 *  cmsShow
 *
 *  Created by Ianna Osborne on 7/8/10.
 *
 */

#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

class FWCastorRecHitProxyBuilder : public FWDigitSetProxyBuilder
{
public:
   FWCastorRecHitProxyBuilder( void ) {}  
   virtual ~FWCastorRecHitProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCastorRecHitProxyBuilder( const FWCastorRecHitProxyBuilder& );
   const FWCastorRecHitProxyBuilder& operator=( const FWCastorRecHitProxyBuilder& );

   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );	
};

void FWCastorRecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
   const CastorRecHitCollection* collection = 0;
   iItem->get( collection );
   if (! collection)
      return;


   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   int index = 0;
   for (std::vector<CastorRecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it)
   {  
      const float* corners = item()->getGeom()->getCorners((*it).detid());
      if (corners == 0) 
         continue;

      std::vector<float> scaledCorners(24);
      fireworks::energyTower3DCorners(corners, (*it).energy() * 10, scaledCorners);

      addBox(boxSet, &scaledCorners[0], iItem->modelInfo(index++).displayProperties());
   }
}

REGISTER_FWPROXYBUILDER( FWCastorRecHitProxyBuilder, CastorRecHitCollection, "Castor RecHit", FWViewType::kISpyBit );
