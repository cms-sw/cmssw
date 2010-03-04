#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "TEveCompound.h"

class FWHBHERecHit3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<HBHERecHit>
{
public:
   FWHBHERecHit3DProxyBuilder(void)
     : m_maxEnergy(0.85)
    {}
  
   virtual ~FWHBHERecHit3DProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   void build(const HBHERecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   Float_t m_maxEnergy;

   // Disable default copy constructor
   FWHBHERecHit3DProxyBuilder(const FWHBHERecHit3DProxyBuilder&);
   // Disable default assignment operator
   const FWHBHERecHit3DProxyBuilder& operator=(const FWHBHERecHit3DProxyBuilder&);
};

void
FWHBHERecHit3DProxyBuilder::build(const HBHERecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   std::vector<TEveVector> corners = item()->getGeom()->getPoints(iData.detid().rawId());
   if( corners.empty() ) {
      return;
   }

   Float_t maxEnergy = m_maxEnergy;
   Float_t energy = iData.energy();
   if(energy > maxEnergy)
   {
     maxEnergy = energy;
   }
   
   Float_t scaleFraction = energy / maxEnergy;
   
   fireworks::drawCaloHit3D(corners, item(), oItemHolder, scaleFraction);
}

REGISTER_FW3DDATAPROXYBUILDER(FWHBHERecHit3DProxyBuilder, HBHERecHit, "HBHE RecHit");
