#include "Fireworks/Calo/plugins/FWCaloRecHitDigitSetProxyBuilder.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"


class FWHORecHitProxyBuilder : public FWCaloRecHitDigitSetProxyBuilder
{
public:
   FWHORecHitProxyBuilder( void ) { invertBox(true); }
   virtual ~FWHORecHitProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWHORecHitProxyBuilder( const FWHORecHitProxyBuilder& );
   const FWHORecHitProxyBuilder& operator=( const FWHORecHitProxyBuilder& );
};

REGISTER_FWPROXYBUILDER( FWHORecHitProxyBuilder, HORecHitCollection, "HO RecHit", FWViewType::kISpyBit );

// AMT scale around center, box is inverted. Scaling and e/et mode added now. Previously used fireworks::energyScaledBox3DCorners().

/*
void
FWHORecHitProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
   const HORecHitCollection* collection = 0;
   iItem->get( collection );

   if( 0 == collection )
   {
      return;
   }
   std::vector<HORecHit>::const_iterator it = collection->begin();
   std::vector<HORecHit>::const_iterator itEnd = collection->end();
   for( ; it != itEnd; ++it )
   {
      if(( *it ).energy() > m_maxEnergy)
         m_maxEnergy = ( *it ).energy();
   }

   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   int index = 0;
   for (std::vector<HORecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it)
   {  
      const float* corners = item()->getGeom()->getCorners((*it).detid());
      std::vector<float> scaledCorners(24);
      if (corners)
         fireworks::energyScaledBox3DCorners(corners, (*it).energy() / m_maxEnergy, scaledCorners, true);

      addBox(boxSet, &scaledCorners[0], iItem->modelInfo(index++).displayProperties());
   }
}
*/
