#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "TEveCompound.h"

class FWCaloRecHit3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<CaloRecHit>
{
public:
   FWCaloRecHit3DProxyBuilder(void) 
    {}
  
   virtual ~FWCaloRecHit3DProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWCaloRecHit3DProxyBuilder(const FWCaloRecHit3DProxyBuilder&);
   // Disable default assignment operator
   const FWCaloRecHit3DProxyBuilder& operator=(const FWCaloRecHit3DProxyBuilder&);

   void build(const CaloRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void
FWCaloRecHit3DProxyBuilder::build(const CaloRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   std::vector<TEveVector> corners = item()->getGeom()->getPoints(iData.detid());
   if( corners.empty() ) {
     return;
   }
   Float_t scale = 10.0; 	// FIXME: The scale should be taken form somewhere else
   Float_t energy = iData.energy();
   Float_t eScale = scale * energy;

   fireworks::drawEcalHit3D(corners, item(), oItemHolder, eScale);
}

REGISTER_FW3DDATAPROXYBUILDER(FWCaloRecHit3DProxyBuilder, CaloRecHit, "Calo RecHit");
