#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TEveCompound.h"

class FWHORecHitProxyBuilder : public FWProxyBuilderBase
{
public:
   FWHORecHitProxyBuilder(void) 
     : m_maxEnergy(1.0)
    {}
  
   virtual ~FWHORecHitProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);

   Float_t m_maxEnergy;

   // Disable default copy constructor
   FWHORecHitProxyBuilder(const FWHORecHitProxyBuilder&);
   // Disable default assignment operator
   const FWHORecHitProxyBuilder& operator=(const FWHORecHitProxyBuilder&);
};

void
FWHORecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
   const HORecHitCollection* collection = 0;
   iItem->get(collection);

   if(0 == collection)
   {
      return;
   }
   std::vector<HORecHit>::const_iterator it = collection->begin();
   std::vector<HORecHit>::const_iterator itEnd = collection->end();
   for(; it != itEnd; ++it)
   {
      if ((*it).energy() > m_maxEnergy)
	m_maxEnergy = (*it).energy();
   }

   unsigned int index = 0;
   for(it = collection->begin(); it != itEnd; ++it, ++index)
   {
      std::vector<TEveVector> corners = iItem->getGeom()->getPoints((*it).detid().rawId());
      if( corners.empty() ) {
	return;
      }

      TEveCompound* compound = createCompound();
      Float_t energy = (*it).energy();
      fireworks::drawEnergyScaledBox3D(corners, energy / m_maxEnergy, compound, this, true );
      setupAddElement(compound, product);
   }
}

REGISTER_FWPROXYBUILDER( FWHORecHitProxyBuilder, HORecHitCollection, "HO RecHit", FWViewType::kISpyBit );
