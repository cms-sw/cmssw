/*
 *  FWCastorRecHitProxyBuilder.cc
 *  cmsShow
 *
 *  Created by Ianna Osborne on 7/8/10.
 *
 */
#include "Fireworks/Calo/plugins/FWCaloRecHitDigitSetProxyBuilder.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

class FWCastorRecHitProxyBuilder : public FWCaloRecHitDigitSetProxyBuilder
{
public:
   FWCastorRecHitProxyBuilder( void ) {}  
   virtual ~FWCastorRecHitProxyBuilder( void ) {}


   virtual float scaleFactor(const FWViewContext* vc) { return 10 * FWCaloRecHitDigitSetProxyBuilder::scaleFactor(vc); } 

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCastorRecHitProxyBuilder( const FWCastorRecHitProxyBuilder& );
   const FWCastorRecHitProxyBuilder& operator=( const FWCastorRecHitProxyBuilder& );
};

REGISTER_FWPROXYBUILDER( FWCastorRecHitProxyBuilder, CastorRecHitCollection, "Castor RecHit", FWViewType::kISpyBit );

// AMT:: scale box round center. Scaleing and e/et added now. Previously used fireworks::energyTower3DCorners();

/*
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
*/
