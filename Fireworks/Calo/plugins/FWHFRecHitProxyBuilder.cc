#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TEveCompound.h"

class FWHFRecHitProxyBuilder : public FWProxyBuilderBase
{
public:
   FWHFRecHitProxyBuilder(void) 
     : m_maxEnergy(5.0)
    {}
  
   virtual ~FWHFRecHitProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);

   Float_t m_maxEnergy;

   // Disable default copy constructor
   FWHFRecHitProxyBuilder(const FWHFRecHitProxyBuilder&);
   // Disable default assignment operator
   const FWHFRecHitProxyBuilder& operator=(const FWHFRecHitProxyBuilder&);
};

void
FWHFRecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
   const HFRecHitCollection* collection = 0;
   iItem->get(collection);

   if(0 == collection)
   {
      return;
   }
   std::vector<HFRecHit>::const_iterator it = collection->begin();
   std::vector<HFRecHit>::const_iterator itEnd = collection->end();
   for(; it != itEnd; ++it)
   {
      if ((*it).energy() > m_maxEnergy)
	m_maxEnergy = (*it).energy();
   }
   
   unsigned int index = 0;
   for(it = collection->begin(); it != itEnd; ++it, ++index)
   {
      Float_t energy = (*it).energy();
      std::vector<TEveVector> corners = iItem->getGeom()->getPoints((*it).detid().rawId());
      if( corners.empty() ) {
	return;
      }

      TEveCompound* compound = createCompound();
      fireworks::drawEnergyScaledBox3D(corners, energy / m_maxEnergy, compound, this, true );
      setupAddElement(compound, product);
   }
}

REGISTER_FWPROXYBUILDER( FWHFRecHitProxyBuilder, HFRecHitCollection, "HF RecHit", FWViewType::kISpyBit );
