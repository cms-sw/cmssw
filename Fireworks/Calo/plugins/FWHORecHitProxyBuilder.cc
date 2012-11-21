#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TEveCompound.h"

class FWHORecHitProxyBuilder : public FWDigitSetProxyBuilder
{
public:
   FWHORecHitProxyBuilder( void ) 
     : m_maxEnergy( 1.0 )
    {}
  
   virtual ~FWHORecHitProxyBuilder( void ) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );

   Float_t m_maxEnergy;

   FWHORecHitProxyBuilder( const FWHORecHitProxyBuilder& );
   const FWHORecHitProxyBuilder& operator=( const FWHORecHitProxyBuilder& );
};

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

REGISTER_FWPROXYBUILDER( FWHORecHitProxyBuilder, HORecHitCollection, "HO RecHit", FWViewType::kISpyBit );
