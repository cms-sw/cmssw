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
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWParameters.h"

#include "TEveTrack.h"
#include "TEveBoxSet.h"

class FWCaloParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<CaloParticle>
{
public:
   FWCaloParticleProxyBuilder( void ) {}
   ~FWCaloParticleProxyBuilder( void ) override {}

   void setItem(const FWEventItem* iItem) override {
      FWProxyBuilderBase::setItem(iItem);
      iItem->getConfig()->assertParam("Point Size", 1l, 3l, 1l);
   }

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWCaloParticleProxyBuilder( const FWCaloParticleProxyBuilder& ) = delete;
   // Disable default assignment operator
   const FWCaloParticleProxyBuilder& operator=( const FWCaloParticleProxyBuilder& ) = delete;

   using FWSimpleProxyBuilderTemplate<CaloParticle>::build;
   void build( const CaloParticle& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;
};

void
FWCaloParticleProxyBuilder::build( const CaloParticle& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
   TEveRecTrack t;
   t.fBeta = 1.0;
   t.fP = TEveVector( iData.px(), iData.py(), iData.pz() );
   t.fV = TEveVector( iData.g4Tracks()[0].trackerSurfacePosition().x(),
                      iData.g4Tracks()[0].trackerSurfacePosition().y(),
                      iData.g4Tracks()[0].trackerSurfacePosition().z() );
   t.fSign = iData.charge();

   TEveTrack* track = new TEveTrack(&t, context().getTrackPropagator());
   if( t.fSign == 0 )
      track->SetLineStyle( 7 );

   track->MakeTrack();
   setupAddElement( track, &oItemHolder );
   TEveBoxSet* boxset = new TEveBoxSet();
   boxset->Reset(TEveBoxSet::kBT_FreeBox, true, 64);
   boxset->UseSingleColor();
   boxset->SetPickable(true);
   for (const auto & c : iData.simClusters())
   {
     auto clusterDetIds = (*c).hits_and_fractions();


     for( auto it = clusterDetIds.begin(), itEnd = clusterDetIds.end();
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

REGISTER_FWPROXYBUILDER( FWCaloParticleProxyBuilder, CaloParticle, "CaloParticles", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
