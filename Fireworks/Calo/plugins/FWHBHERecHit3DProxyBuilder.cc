#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TEveCompound.h"
#include "TEveManager.h"

class FWHBHERecHit3DProxyBuilder : public FW3DDataProxyBuilder
{
public:
   FWHBHERecHit3DProxyBuilder(void)
     : m_maxEnergy(0.85)
    {}
  
   virtual ~FWHBHERecHit3DProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   Float_t m_maxEnergy;

   // Disable default copy constructor
   FWHBHERecHit3DProxyBuilder(const FWHBHERecHit3DProxyBuilder&);
   // Disable default assignment operator
   const FWHBHERecHit3DProxyBuilder& operator=(const FWHBHERecHit3DProxyBuilder&);
};

void
FWHBHERecHit3DProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList = new TEveElementList(iItem->name().c_str(), "hbheRechits", true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

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
   Color_t color = iItem->defaultDisplayProperties().color();
   
   unsigned int index = 0;
   for(it = collection->begin(); it != itEnd; ++it, ++index)
   {
      Float_t energy = (*it).energy();

      std::stringstream s;
      s << "HBHE RecHit " << index << ", energy: " << energy << " GeV";

      TEveCompound* compund = new TEveCompound("hbhe compound", s.str().c_str());
      compund->OpenCompound();
      tList->AddElement(compund);
      
      std::vector<TEveVector> corners = iItem->getGeom()->getPoints((*it).detid().rawId());
      if( corners.empty() ) {
	return;
      }
   
      fireworks::drawEnergyScaledBox3D(corners, energy / m_maxEnergy, color, *compund);
   }
}

REGISTER_FW3DDATAPROXYBUILDER(FWHBHERecHit3DProxyBuilder, HBHERecHitCollection, "HBHE RecHit");
