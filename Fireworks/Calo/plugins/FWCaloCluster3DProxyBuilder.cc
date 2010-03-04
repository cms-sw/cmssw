#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "TEveCompound.h"

class FWCaloCluster3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::CaloCluster>
{
public:
   FWCaloCluster3DProxyBuilder(void) 
    {}
  
   virtual ~FWCaloCluster3DProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloCluster3DProxyBuilder(const FWCaloCluster3DProxyBuilder&); 			// stop default
   const FWCaloCluster3DProxyBuilder& operator=(const FWCaloCluster3DProxyBuilder&); 	// stop default

   void build(const reco::CaloCluster& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void
FWCaloCluster3DProxyBuilder::build(const reco::CaloCluster& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   std::vector<std::pair<DetId, float> > clusterDetIds = iData.hitsAndFractions ();
   for(std::vector<std::pair<DetId, float> >::iterator id = clusterDetIds.begin (), idend = clusterDetIds.end ();
       id != idend; ++id)
   {
      std::vector<TEveVector> corners = item()->getGeom()->getPoints((*id).first);
      if( corners.empty() ) {
	 continue;
      }
      Float_t scale = 10.0; 	// FIXME: The scale should be taken form somewhere else
      Float_t energy = (*id).second;
      Float_t eScale = scale * energy;

      fireworks::drawEcalHit3D(corners, item(), oItemHolder, eScale);
   }
}

REGISTER_FW3DDATAPROXYBUILDER(FWCaloCluster3DProxyBuilder, reco::CaloCluster, "Calo Cluster");
