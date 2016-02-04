#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TEveCompound.h"

class FWHFRecHitProxyBuilder : public FWDigitSetProxyBuilder
{
public:
   FWHFRecHitProxyBuilder( void ) 
     : m_maxEnergy( 5.0 )
    {}
  
   virtual ~FWHFRecHitProxyBuilder( void ) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );

   Float_t m_maxEnergy;

   FWHFRecHitProxyBuilder( const FWHFRecHitProxyBuilder& );
   const FWHFRecHitProxyBuilder& operator=( const FWHFRecHitProxyBuilder& );
};

void
FWHFRecHitProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
   const HFRecHitCollection* collection = 0;
   iItem->get( collection );

   if( 0 == collection )
   {
      return;
   }

   std::vector<HFRecHit>::const_iterator it = collection->begin();
   std::vector<HFRecHit>::const_iterator itEnd = collection->end();
   for( ; it != itEnd; ++it )
   {
      if(( *it ).energy() > m_maxEnergy )
         m_maxEnergy = ( *it ).energy();
   }

   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   int index = 0;
   for (std::vector<HFRecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it)
   {  
      unsigned int rawid = ( *it ).detid().rawId();
      if( ! context().getGeom()->contains( rawid ))
      {
         fwLog( fwlog::kInfo ) << "FWHFRecHitProxyBuilder cannot get geometry for DetId: "
                               << rawid << ". Ignored.\n";
      }
      const float* corners = context().getGeom()->getCorners( rawid );

      std::vector<float> scaledCorners(24);
      if (corners)
         fireworks::energyScaledBox3DCorners(corners, (*it).energy() / m_maxEnergy, scaledCorners, true);

      addBox(boxSet, &scaledCorners[0], iItem->modelInfo(index++).displayProperties());
   }
}

REGISTER_FWPROXYBUILDER( FWHFRecHitProxyBuilder, HFRecHitCollection, "HF RecHit", FWViewType::kISpyBit );
