#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Fireworks/Calo/plugins/FWCaloRecHitDigitSetProxyBuilder.h"

class FWHFRecHitProxyBuilder : public FWCaloRecHitDigitSetProxyBuilder
{
public:
   FWHFRecHitProxyBuilder( void ) {invertBox(true); }
   virtual ~FWHFRecHitProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWHFRecHitProxyBuilder( const FWHFRecHitProxyBuilder& );
   const FWHFRecHitProxyBuilder& operator=( const FWHFRecHitProxyBuilder& );
};


REGISTER_FWPROXYBUILDER( FWHFRecHitProxyBuilder, HFRecHitCollection, "HF RecHit", FWViewType::kISpyBit );

// AMT: Reflect box. Previously used energyScaledBox3DCorners(). Scaling and e/et mode added now.

/*
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
*/
