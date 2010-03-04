#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "TEveCompound.h"

class FWHFRecHit3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<HFRecHit>
{
public:
   FWHFRecHit3DProxyBuilder(void) 
     : m_maxEnergy(5.0)
    {}
  
   virtual ~FWHFRecHit3DProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   void build(const HFRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   Float_t m_maxEnergy;

   // Disable default copy constructor
   FWHFRecHit3DProxyBuilder(const FWHFRecHit3DProxyBuilder&);
   // Disable default assignment operator
   const FWHFRecHit3DProxyBuilder& operator=(const FWHFRecHit3DProxyBuilder&);
};

void
FWHFRecHit3DProxyBuilder::build(const HFRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const
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

REGISTER_FW3DDATAPROXYBUILDER(FWHFRecHit3DProxyBuilder, HFRecHit, "HF RecHit");
