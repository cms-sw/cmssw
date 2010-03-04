#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "TEveCompound.h"

class FWEcalRecHit3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<EcalRecHit>
{
public:
   FWEcalRecHit3DProxyBuilder(void) 
    {}
  
   virtual ~FWEcalRecHit3DProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWEcalRecHit3DProxyBuilder(const FWEcalRecHit3DProxyBuilder&);
   // Disable default assignment operator
   const FWEcalRecHit3DProxyBuilder& operator=(const FWEcalRecHit3DProxyBuilder&);

   void build(const EcalRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void
FWEcalRecHit3DProxyBuilder::build(const EcalRecHit& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   std::vector<TEveVector> corners = item()->getGeom()->getPoints(iData.id());
   if( corners.empty() ) {
     return;
   }
   Float_t scale = 10.0; 	// FIXME: The scale should be taken form somewhere else
   Float_t energy = iData.energy();
   Float_t eScale = scale * energy;

   fireworks::drawEcalHit3D(corners, item(), oItemHolder, eScale);
}

REGISTER_FW3DDATAPROXYBUILDER(FWEcalRecHit3DProxyBuilder, EcalRecHit, "Ecal RecHit");
