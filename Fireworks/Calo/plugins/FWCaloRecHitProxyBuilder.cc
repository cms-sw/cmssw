#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "TEveCompound.h"

class FWCaloRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<CaloRecHit>
{
public:
   FWCaloRecHitProxyBuilder(void) 
    {}
  
   virtual ~FWCaloRecHitProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWCaloRecHitProxyBuilder(const FWCaloRecHitProxyBuilder&);
   // Disable default assignment operator
   const FWCaloRecHitProxyBuilder& operator=(const FWCaloRecHitProxyBuilder&);

   void build(const CaloRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void
FWCaloRecHitProxyBuilder::build(const CaloRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   std::vector<TEveVector> corners = item()->getGeom()->getPoints(iData.detid());
   if( corners.empty() ) {
     return;
   }
   Float_t scale = 10.0; 	// FIXME: The scale should be taken form somewhere else

   fireworks::drawEnergyTower3D(corners, iData.energy() * scale, oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWCaloRecHitProxyBuilder, CaloRecHit, "Calo RecHit", FWViewType::kISpy );
