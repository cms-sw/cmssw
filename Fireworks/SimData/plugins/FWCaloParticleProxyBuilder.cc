/*
 *  FWCaloParticleProxyBuilder.cc
 *  FWorks
 *
 *  Created by Marco Rovere 13/09/2018
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "FWCore/Common/interface/EventBase.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWParameters.h"

#include "TGeoBBox.h"
#include "TGeoXtru.h"

#include "TEveGeoShape.h"
#include "TEveTrack.h"
#include "TEveBoxSet.h"

class FWCaloParticleProxyBuilder : public FWProxyBuilderBase
{
public:
   FWCaloParticleProxyBuilder( void ) {}
   ~FWCaloParticleProxyBuilder( void ) override {}

   REGISTER_PROXYBUILDER_METHODS();
private:
   // Disable default copy constructor
   FWCaloParticleProxyBuilder( const FWCaloParticleProxyBuilder& ) = delete;
   // Disable default assignment operator
   const FWCaloParticleProxyBuilder& operator=( const FWCaloParticleProxyBuilder& ) = delete;

   void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* ) override;
};

void
FWCaloParticleProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* ){
   const CaloParticleCollection* collection = nullptr;
   iItem->get( collection );

   if( collection == nullptr ) return;

   for( const auto & iData : *collection )
   {
      TEveBoxSet* boxset = new TEveBoxSet();
      boxset->Reset(TEveBoxSet::kBT_FreeBox, true, 64);
      boxset->UseSingleColor();
      boxset->SetPickable(true);

      for (const auto & c : iData.simClusters())
      {
         for( const auto & it : (*c).hits_and_fractions() )
         {  
            const float* corners = item()->getGeom()->getCorners( it.first );
            const float* parameters = item()->getGeom()->getParameters( it.first );
            const float* shapes = item()->getGeom()->getShapePars(it.first);

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
      }
      boxset->RefitPlex();
      setupAddElement(boxset, product);
   }
}

REGISTER_FWPROXYBUILDER( FWCaloParticleProxyBuilder, CaloParticleCollection, "CaloFTW", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
