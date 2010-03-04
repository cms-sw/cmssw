#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "TEveCompound.h"

class FWHORecHit3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<HORecHit>
{
public:
   FWHORecHit3DProxyBuilder(void) 
     : m_maxEnergy(1.0)
    {}
  
   virtual ~FWHORecHit3DProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   void build(const HORecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   Float_t m_maxEnergy;

   // Disable default copy constructor
   FWHORecHit3DProxyBuilder(const FWHORecHit3DProxyBuilder&);
   // Disable default assignment operator
   const FWHORecHit3DProxyBuilder& operator=(const FWHORecHit3DProxyBuilder&);
};

void
FWHORecHit3DProxyBuilder::build(const HORecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const
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

REGISTER_FW3DDATAPROXYBUILDER(FWHORecHit3DProxyBuilder, HORecHit, "HO RecHit");
