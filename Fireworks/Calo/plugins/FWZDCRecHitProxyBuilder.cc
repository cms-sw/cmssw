/*
 *  FWZDCRecHitProxyBuilder.cc
 *  cmsShow
 *
 *  Created by Ianna Osborne on 7/8/10.
 *
 */
#include "Fireworks/Calo/plugins/FWCaloRecHitDigitSetProxyBuilder.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/Common/interface/SortedCollection.h"


class FWZDCRecHitProxyBuilder :  public FWCaloRecHitDigitSetProxyBuilder
{
public:
   FWZDCRecHitProxyBuilder( void ) {}  
   virtual ~FWZDCRecHitProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWZDCRecHitProxyBuilder( const FWZDCRecHitProxyBuilder& );
   const FWZDCRecHitProxyBuilder& operator=( const FWZDCRecHitProxyBuilder& );	
};


REGISTER_FWPROXYBUILDER( FWZDCRecHitProxyBuilder,  edm::SortedCollection<ZDCRecHit> , "ZDC RecHit", FWViewType::kISpyBit );

// AMT scale box round center. Scaling and e/et mode added now. Previusly used energyTower3DCorners().

/*
void FWZDCRecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
   const edm::SortedCollection<ZDCRecHit> *collection = 0;
   iItem->get( collection );
   if (! collection)
      return;


   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   boxSet->SetAntiFlick(kTRUE);
   int index = 0;
   for (std::vector<ZDCRecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it)
   {  
      const float* corners = item()->getGeom()->getCorners((*it).detid());

      std::vector<float> scaledCorners(24);
      if (corners != 0) {
         fireworks::energyTower3DCorners(corners, (*it).energy(), scaledCorners);
         // Invert the normals:
         // for (int i = 0; i < 12; ++i)
         //    std::swap(scaledCorners[i], scaledCorners[i+12]);
      }

      addBox(boxSet, &scaledCorners[0], iItem->modelInfo(index++).displayProperties());
   }
   }*/
