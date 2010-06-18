#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TEveCompound.h"

class FWHBHERecHitProxyBuilder : public FWProxyBuilderBase
{
public:
   FWHBHERecHitProxyBuilder(void)
     : m_maxEnergy(0.85)
    {}
  
   virtual ~FWHBHERecHitProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);

   Float_t m_maxEnergy;

   // Disable default copy constructor
   FWHBHERecHitProxyBuilder(const FWHBHERecHitProxyBuilder&);
   // Disable default assignment operator
   const FWHBHERecHitProxyBuilder& operator=(const FWHBHERecHitProxyBuilder&);
};

void
FWHBHERecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
   const HBHERecHitCollection* collection = 0;
   iItem->get(collection);

   if(0 == collection)
   {
      return;
   }
   std::vector<HBHERecHit>::const_iterator it = collection->begin();
   std::vector<HBHERecHit>::const_iterator itEnd = collection->end();
   for(; it != itEnd; ++it)
   {
      if ((*it).energy() > m_maxEnergy)
         m_maxEnergy = (*it).energy();
   }
   
   unsigned int index = 0;
   for(it = collection->begin(); it != itEnd; ++it, ++index)
   {
      Float_t energy = (*it).energy();

      TEveCompound* compound = createCompound();
      setupAddElement(compound, product);

      std::vector<TEveVector> corners = iItem->getGeom()->getPoints((*it).detid());
      if( corners.empty() ) {
         continue;
      }

      fireworks::drawEnergyScaledBox3D(corners, energy / m_maxEnergy, compound, this, true );
   }
}

REGISTER_FWPROXYBUILDER( FWHBHERecHitProxyBuilder, HBHERecHitCollection, "HBHE RecHit", FWViewType::kISpyBit );
