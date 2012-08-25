/*
 *  FWEcalRecHitProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 5/28/10.
 *
 */
#include "TEveBoxSet.h"
#include "TEveChunkManager.h"
#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class FWEcalRecHitProxyBuilder : public FWDigitSetProxyBuilder
{
public:
   FWEcalRecHitProxyBuilder():FWDigitSetProxyBuilder(), m_plotEt(true) {}
   virtual ~FWEcalRecHitProxyBuilder() {}
 
   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);
	
   REGISTER_PROXYBUILDER_METHODS();
	
private:
   FWEcalRecHitProxyBuilder( const FWEcalRecHitProxyBuilder& );
   const FWEcalRecHitProxyBuilder& operator=( const FWEcalRecHitProxyBuilder& );

   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );	

   bool m_plotEt;
};

//______________________________________________________________________________


void
FWEcalRecHitProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
 
   if (m_plotEt != vc->getEnergyScale()->getPlotEt() )
   {
      m_plotEt = !m_plotEt;

      const EcalRecHitCollection* collection = 0;
      item()->get( collection );
      if (! collection)
         return;

      int index = 0;
      std::vector<float> scaledCorners(24);
      for (std::vector<EcalRecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it, ++index)
      {
         const float* corners = item()->getGeom()->getCorners((*it).detid());
         if (corners == 0) 
            continue;

         Float_t scale = 10.0;
         bool reflect = false;
         if (EcalSubdetector( (*it).detid().subdetId() ) == EcalPreshower)
         {
            scale = 1000.0; 	// FIXME: The scale should be taken form somewhere else
            reflect = corners[2] < 0;
         }

         FWDigitSetProxyBuilder::BFreeBox_t* b = (FWDigitSetProxyBuilder::BFreeBox_t*)getBoxSet()->GetPlex()->Atom(index);
         /*
           printf("--------------------scale product \n");
           for (int i = 0; i < 8 ; ++i)
           printf("[%f %f %f ]\n",b->fVertices[i][0], b->fVertices[i][1],b->fVertices[i][2] );
         */

         if (m_plotEt)
            fireworks::etTower3DCorners(corners, (*it).energy() * scale,  scaledCorners, reflect);
         else
            fireworks::energyTower3DCorners(corners, (*it).energy() * scale,  scaledCorners, reflect);

         memcpy(b->fVertices, &scaledCorners[0], sizeof(b->fVertices));

         /*
           printf("after \n");
           for (int i = 0; i < 8 ; ++i)
           printf("[%f %f %f ]\n",b->fVertices[i][0], b->fVertices[i][1],b->fVertices[i][2] );
         */
      }
      getBoxSet()->ElementChanged();
   }
}

//______________________________________________________________________________

void FWEcalRecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext* vc)
{
   m_plotEt = vc->getEnergyScale()->getPlotEt();

   const EcalRecHitCollection* collection = 0;
   iItem->get( collection );
   if (! collection)
      return;

   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   std::vector<float> scaledCorners(24);
   int index = 0;
   for (std::vector<EcalRecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it)
   {
      const float* corners = item()->getGeom()->getCorners((*it).detid());
      if (corners == 0) 
         continue;

      Float_t scale = 10.0;
      bool reflect = false;
      if (EcalSubdetector( (*it).detid().subdetId() ) == EcalPreshower)
      {
         scale = 1000.0; 	// FIXME: The scale should be taken form somewhere else
         reflect = corners[2] < 0;
      }

      if (m_plotEt)
         fireworks::energyTower3DCorners(corners, (*it).energy() * scale,  scaledCorners, reflect);
      else
         fireworks::energyTower3DCorners(corners, (*it).energy() * scale,  scaledCorners, reflect);

      addBox(boxSet, &scaledCorners[0], iItem->modelInfo(index++).displayProperties());
   }
}

REGISTER_FWPROXYBUILDER( FWEcalRecHitProxyBuilder, EcalRecHitCollection, "Ecal RecHit", FWViewType::kISpyBit );
