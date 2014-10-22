#include "Fireworks/Calo/plugins/FWCaloRecHitDigitSetProxyBuilder.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

class FWHGCEERecHitProxyBuilder : public FWCaloRecHitDigitSetProxyBuilder
{
public:
   FWHGCEERecHitProxyBuilder( void ) { invertBox(true); }
   virtual ~FWHGCEERecHitProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWHGCEERecHitProxyBuilder( const FWHGCEERecHitProxyBuilder& );
   const FWHGCEERecHitProxyBuilder& operator=( const FWHGCEERecHitProxyBuilder& );
};

REGISTER_FWPROXYBUILDER( FWHGCEERecHitProxyBuilder, HGCRecHitCollection, "HGCEE RecHit", FWViewType::kISpyBit );

// AMT: Refelct box. Previously used energyScaledBox3DCorners()

/*
void
FWHGCEERecHitProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* vc)
{
   m_plotEt = vc->getEnergyScale()->getPlotEt();

   const HBHERecHitCollection* collection = 0;
   iItem->get( collection );

   if( 0 == collection )
   {
      return;
   }
   std::vector<HBHERecHit>::const_iterator it = collection->begin();
   std::vector<HBHERecHit>::const_iterator itEnd = collection->end();
   std::vector<float> scaledCorners(24);

   for( ; it != itEnd; ++it )
   {
      if(( *it ).energy() > m_maxEnergy )
         m_maxEnergy = ( *it ).energy();
   }

   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   int index = 0;
   for (std::vector<HBHERecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it)
   {  
      const float* corners = context().getGeom()->getCorners((*it).detid());
      if (corners)
      {
         if (m_plotEt)
            fireworks::etScaledBox3DCorners(corners, (*it).energy(), m_maxEnergy, scaledCorners, true);
         else
            fireworks::energyScaledBox3DCorners(corners, (*it).energy() / m_maxEnergy, scaledCorners, true);
      }
      addBox(boxSet, &scaledCorners[0], iItem->modelInfo(index++).displayProperties());
   }
}
*/
