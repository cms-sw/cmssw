#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TEveCompound.h"
#include "TEveManager.h"

class FWHFRecHit3DProxyBuilder : public FW3DDataProxyBuilder
{
public:
   FWHFRecHit3DProxyBuilder(void) 
     : m_maxEnergy(5.0)
    {}
  
   virtual ~FWHFRecHit3DProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   Float_t m_maxEnergy;

   // Disable default copy constructor
   FWHFRecHit3DProxyBuilder(const FWHFRecHit3DProxyBuilder&);
   // Disable default assignment operator
   const FWHFRecHit3DProxyBuilder& operator=(const FWHFRecHit3DProxyBuilder&);
};

void
FWHFRecHit3DProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList = new TEveElementList(iItem->name().c_str(), "hfRechits", true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

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
   Color_t color = iItem->defaultDisplayProperties().color();
   
   unsigned int index = 0;
   for(it = collection->begin(); it != itEnd; ++it, ++index)
   {
      Float_t energy = (*it).energy();

      std::stringstream s;
      s << "HF RecHit " << index << ", energy: " << energy << " GeV";

      TEveCompound* compund = new TEveCompound("hf compound", s.str().c_str());
      compund->OpenCompound();
      tList->AddElement(compund);
      
      std::vector<TEveVector> corners = iItem->getGeom()->getPoints((*it).detid().rawId());
      if( corners.empty() ) {
	return;
      }
   
      fireworks::drawEnergyScaledBox3D(corners, energy / m_maxEnergy, color, *compund);
   }
}

REGISTER_FW3DDATAPROXYBUILDER(FWHFRecHit3DProxyBuilder, HFRecHitCollection, "HF RecHit");
