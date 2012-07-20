/*
 *  FWEcalRecHitProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 5/28/10.
 *
 */
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Fireworks/Calo/plugins/FWCaloRecHitDigitSetProxyBuilder.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class FWEcalRecHitProxyBuilder : public FWCaloRecHitDigitSetProxyBuilder
{
public:
   FWEcalRecHitProxyBuilder() {}
   virtual ~FWEcalRecHitProxyBuilder() {}
 
   virtual void viewContextBoxScale( const float* corners, float scale, bool plotEt, std::vector<float>& scaledCorners, const CaloRecHit*);
	
   REGISTER_PROXYBUILDER_METHODS();
	
private:
   FWEcalRecHitProxyBuilder( const FWEcalRecHitProxyBuilder& );
   const FWEcalRecHitProxyBuilder& operator=( const FWEcalRecHitProxyBuilder& );
};


void FWEcalRecHitProxyBuilder::viewContextBoxScale( const float* corners, float scale, bool plotEt, std::vector<float>& scaledCorners, const CaloRecHit* hit)
{ 
   invertBox((EcalSubdetector( hit->detid().subdetId() ) == EcalPreshower) && (corners[2] < 0));
   FWCaloRecHitDigitSetProxyBuilder::viewContextBoxScale(corners, scale, plotEt, scaledCorners, hit );
}


REGISTER_FWPROXYBUILDER( FWEcalRecHitProxyBuilder, EcalRecHitCollection, "Ecal RecHit", FWViewType::kISpyBit );

// AMT: Scale box round cener. Prviousy used fireworks::energyTower3DCorners().
// Why differnt scale factor in  EcalPreShower ???


/*
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
*/
