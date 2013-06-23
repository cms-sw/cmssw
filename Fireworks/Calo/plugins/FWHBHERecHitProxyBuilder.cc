#include "TEveCompound.h"
#include "TEveBoxSet.h"

#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

class FWHBHERecHitProxyBuilder : public FWDigitSetProxyBuilder
{
public:
   FWHBHERecHitProxyBuilder( void )
      : m_maxEnergy( 0.85 ), m_plotEt(true)
    {}
  
   virtual ~FWHBHERecHitProxyBuilder( void ) 
    {}

   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );

   Float_t m_maxEnergy;
   bool m_plotEt;

   FWHBHERecHitProxyBuilder( const FWHBHERecHitProxyBuilder& );
   const FWHBHERecHitProxyBuilder& operator=( const FWHBHERecHitProxyBuilder& );
};

void
FWHBHERecHitProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   if (m_plotEt != vc->getEnergyScale()->getPlotEt() )
   {
      m_plotEt = !m_plotEt;

      const HBHERecHitCollection* collection = 0;
      item()->get( collection );
      if (! collection)
         return;

      int index = 0;
      std::vector<float> scaledCorners(24);
      for (std::vector<HBHERecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it, ++index)
      {
         const float* corners = item()->getGeom()->getCorners((*it).detid());
         if (corners == 0) 
            continue;
         FWDigitSetProxyBuilder::BFreeBox_t* b = (FWDigitSetProxyBuilder::BFreeBox_t*)getBoxSet()->GetPlex()->Atom(index);

         /*          
         printf("--------------------scale product \n");
         for (int i = 0; i < 8 ; ++i)
            printf("[%f %f %f ]\n",b->fVertices[i][0], b->fVertices[i][1],b->fVertices[i][2] );
         */

         if (m_plotEt)
            fireworks::etScaledBox3DCorners(corners, (*it).energy(),  m_maxEnergy, scaledCorners, true);
         else
            fireworks::energyScaledBox3DCorners(corners, (*it).energy() / m_maxEnergy, scaledCorners, true);
         
         /*
         printf("after \n");
         for (int i = 0; i < 8 ; ++i)
            printf("[%f %f %f ]\n",b->fVertices[i][0], b->fVertices[i][1],b->fVertices[i][2] );        
         */
         memcpy(b->fVertices, &scaledCorners[0], sizeof(b->fVertices));

      }
      getBoxSet()->ElementChanged();
   }
}

//______________________________________________________________________________

void
FWHBHERecHitProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* vc)
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

REGISTER_FWPROXYBUILDER( FWHBHERecHitProxyBuilder, HBHERecHitCollection, "HBHE RecHit", FWViewType::kISpyBit );
