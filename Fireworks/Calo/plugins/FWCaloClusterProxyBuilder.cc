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
   ~FWCaloClusterProxyBuilder( void ) override {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloClusterProxyBuilder( const FWCaloClusterProxyBuilder& ) = delete; 			// stop default
   const FWCaloClusterProxyBuilder& operator=( const FWCaloClusterProxyBuilder& ) = delete; 	// stop default

   using FWSimpleProxyBuilderTemplate<reco::CaloCluster>::build;
   void build( const reco::CaloCluster& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;
};

void
FWCaloClusterProxyBuilder::build( const reco::CaloCluster& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
   std::vector<std::pair<DetId, float> > clusterDetIds = iData.hitsAndFractions();
   
   TEveBoxSet* boxset = new TEveBoxSet();
   boxset->Reset(TEveBoxSet::kBT_FreeBox, true, 64);
   boxset->UseSingleColor();
   boxset->SetPickable(true);

   for( std::vector<std::pair<DetId, float> >::iterator it = clusterDetIds.begin(), itEnd = clusterDetIds.end();
        it != itEnd; ++it )
   {
      const float* corners = item()->getGeom()->getCorners( it->first );
      const float* parameters = item()->getGeom()->getParameters( it->first );
      const float* shapes = item()->getGeom()->getShapePars(it->first);

      if( corners == nullptr || parameters == nullptr || shapes == nullptr ) {
         continue;
      }

#if 0
      const int total_points = parameters[0];
      const int total_vertices = 3*total_points;
#else // using broken boxes(half hexagon) until there's support for hexagons in TEveBoxSet
      const int total_points = 4;
      const int total_vertices = 3*total_points;

      const float thickness = shapes[3];

      std::vector<float> pnts(24);
      for(int i = 0; i < total_points; ++i){
         pnts[i*3+0] = corners[i*3];
         pnts[i*3+1] = corners[i*3+1];
         pnts[i*3+2] = corners[i*3+2];

         pnts[(i*3+0)+total_vertices] = corners[i*3];
         pnts[(i*3+1)+total_vertices] = corners[i*3+1];
         pnts[(i*3+2)+total_vertices] = corners[i*3+2]+thickness;
      }
      boxset->AddBox( &pnts[0]);
#endif
   }

   boxset->RefitPlex();
   setupAddElement(boxset, &oItemHolder);
}

REGISTER_FWPROXYBUILDER( FWCaloClusterProxyBuilder, reco::CaloCluster, "Calo Cluster", FWViewType::kISpyBit );
