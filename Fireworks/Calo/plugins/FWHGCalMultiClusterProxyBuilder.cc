#include "TEveBoxSet.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

class FWHGCalMultiClusterProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::HGCalMultiCluster>
{
public:
   FWHGCalMultiClusterProxyBuilder( void ) {}
   ~FWHGCalMultiClusterProxyBuilder( void ) override {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWHGCalMultiClusterProxyBuilder( const FWHGCalMultiClusterProxyBuilder& ) = delete; 			// stop default
   const FWHGCalMultiClusterProxyBuilder& operator=( const FWHGCalMultiClusterProxyBuilder& ) = delete; 	// stop default

   using FWSimpleProxyBuilderTemplate<reco::HGCalMultiCluster>::build;
   void build( const reco::HGCalMultiCluster& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;
};

void
FWHGCalMultiClusterProxyBuilder::build( const reco::HGCalMultiCluster& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
  const auto & clusters = iData.clusters();

  TEveBoxSet* boxset = new TEveBoxSet();
  boxset->Reset(TEveBoxSet::kBT_FreeBox, true, 64);
  boxset->UseSingleColor();
  boxset->SetPickable(true);
  for (const auto & c : clusters)
    {
      std::vector<std::pair<DetId, float> > clusterDetIds = c->hitsAndFractions();


      for( std::vector<std::pair<DetId, float> >::iterator it = clusterDetIds.begin(), itEnd = clusterDetIds.end();
	   it != itEnd; ++it )
	{
	  const float* corners = item()->getGeom()->getCorners( (*it).first );
	  if( corners == nullptr ) {
	    continue;
	  }
	  std::vector<float> pnts(24);
	  fireworks::energyTower3DCorners(corners, (*it).second, pnts);
	  boxset->AddBox( &pnts[0]);
	}
    }
   boxset->RefitPlex();
   setupAddElement(boxset, &oItemHolder);
}

REGISTER_FWPROXYBUILDER( FWHGCalMultiClusterProxyBuilder, reco::HGCalMultiCluster, "HGCal MultiCluster", FWViewType::kISpyBit );
