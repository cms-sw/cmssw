#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

class FWPRCaloTowerProxyBuilder : public FWDigitSetProxyBuilder
{
public:
   FWPRCaloTowerProxyBuilder( void ) {} 
   virtual ~FWPRCaloTowerProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPRCaloTowerProxyBuilder( const FWPRCaloTowerProxyBuilder& ); 			// stop default
   const FWPRCaloTowerProxyBuilder& operator=( const FWPRCaloTowerProxyBuilder& ); 	// stop default

   using FWDigitSetProxyBuilder::build;
   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );	
};


void FWPRCaloTowerProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
   const CaloTowerCollection* collection = 0;
   iItem->get( collection );
   if (! collection)
      return;


   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   int index = 0;
   for (std::vector<CaloTower>::const_iterator it = collection->begin() ; it != collection->end(); ++it)
   {  
      const float* corners = item()->getGeom()->getCorners((*it).id().rawId());
      if (corners == 0) 
         continue;

      std::vector<float> scaledCorners(24);
      fireworks::energyTower3DCorners(corners, (*it).et(), scaledCorners);

     addBox(boxSet, &scaledCorners[0], iItem->modelInfo(index++).displayProperties());

   }
} 

REGISTER_FWPROXYBUILDER( FWPRCaloTowerProxyBuilder, CaloTowerCollection, "PRCaloTower", FWViewType::kISpyBit );
