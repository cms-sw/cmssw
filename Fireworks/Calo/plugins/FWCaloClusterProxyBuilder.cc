#include "TEveBoxSet.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

class FWCaloClusterProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::CaloCluster>
{
public:
   FWCaloClusterProxyBuilder( void ) {}  
   virtual ~FWCaloClusterProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloClusterProxyBuilder( const FWCaloClusterProxyBuilder& ); 			// stop default
   const FWCaloClusterProxyBuilder& operator=( const FWCaloClusterProxyBuilder& ); 	// stop default

   void build( const reco::CaloCluster& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWCaloClusterProxyBuilder::build( const reco::CaloCluster& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
   std::vector<std::pair<DetId, float> > clusterDetIds = iData.hitsAndFractions();
   
   TEveBoxSet* boxset = new TEveBoxSet();
   boxset->Reset(TEveBoxSet::kBT_FreeBox, true, 64);
   boxset->UseSingleColor();
   boxset->SetPickable(1);

   for( std::vector<std::pair<DetId, float> >::iterator it = clusterDetIds.begin(), itEnd = clusterDetIds.end();
        it != itEnd; ++it )
   {
      const float* corners = item()->getGeom()->getCorners( (*it).first );
      if( corners == 0 ) {
         continue;
      }
      std::vector<float> pnts(24);    
      fireworks::energyTower3DCorners(corners, (*it).second, pnts);
      boxset->AddBox( &pnts[0]);
   }

   boxset->RefitPlex();
   setupAddElement(boxset, &oItemHolder);
}

REGISTER_FWPROXYBUILDER( FWCaloClusterProxyBuilder, reco::CaloCluster, "Calo Cluster", FWViewType::kISpyBit );
