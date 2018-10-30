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

#if 1
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

   using FWProxyBuilderBase::build;
   void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* ) override;
};

void
FWCaloParticleProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* ){
   const CaloParticleCollection* collection = nullptr;
   iItem->get( collection );

   if( nullptr == collection )
   {
      return;
   }
   
   const edm::EventBase *event = item()->getEvent();
   // hitmap
   std::map<DetId, const HGCRecHit*> hitmap;
   // max detected energy
   float maxEnergy(1e-5f);

   edm::Handle<HGCRecHitCollection> recHitHandleEE;
   edm::Handle<HGCRecHitCollection> recHitHandleFH;
   edm::Handle<HGCRecHitCollection> recHitHandleBH;   

   event->getByLabel( edm::InputTag( "HGCalRecHit", "HGCEERecHits" ), recHitHandleEE );
   event->getByLabel( edm::InputTag( "HGCalRecHit", "HGCHEFRecHits" ), recHitHandleFH );
   event->getByLabel( edm::InputTag( "HGCalRecHit", "HGCHEBRecHits" ), recHitHandleBH );

   const auto& rechitsEE = *recHitHandleEE;
   const auto& rechitsFH = *recHitHandleFH;
   const auto& rechitsBH = *recHitHandleBH;

   for (unsigned int i = 0; i < rechitsEE.size(); ++i) {
      hitmap[rechitsEE[i].detid().rawId()] = &rechitsEE[i];
      maxEnergy = fmax(maxEnergy, rechitsEE[i].energy());
   }
   for (unsigned int i = 0; i < rechitsFH.size(); ++i) {
      hitmap[rechitsFH[i].detid().rawId()] = &rechitsFH[i];
      maxEnergy = fmax(maxEnergy, rechitsFH[i].energy());   
   }
   for (unsigned int i = 0; i < rechitsBH.size(); ++i) {
      hitmap[rechitsBH[i].detid().rawId()] = &rechitsBH[i];
      maxEnergy = fmax(maxEnergy, rechitsBH[i].energy());
   }
   
   for( std::vector<CaloParticle>::const_iterator iData = collection->begin(), end = collection->end(); iData != end; ++iData )
   {
      for (const auto & c : iData->simClusters())
      {
         auto clusterDetIds = (*c).hits_and_fractions();

         for( auto it = clusterDetIds.begin(), itEnd = clusterDetIds.end();
            it != itEnd; ++it )
         {
            if(hitmap.find(it->first) == hitmap.end()) continue;
#if 0
            TEveGeoShape* shape = item()->getGeom()->getHGCalEveShape(it->first);
#else
            TEveGeoShape* shape = new TEveGeoShape(TString::Format("CaloParticleRecHit Id=%u", it->first));
            
            const float* corners = item()->getGeom()->getCorners( it->first );
            if( corners == nullptr ) {
               continue;
            }

            float dx = fabs(corners[6]  - corners[0]);
            float dy = fabs(corners[4]  - corners[1]);
            float dz = fabs(corners[14] - corners[2]);

            TGeoShape* geoShape = new TGeoBBox( dx, dy, dz );
            shape->SetShape( geoShape );

            double array[16] = { 1., 0., 0., 0.,
                  0., 1., 0., 0.,
                  0., 0., 1., 0.,
                  (corners[6] + corners[0])*0.5f, (corners[4] + corners[1])*0.5f, (corners[14] + corners[2])*0.5f, 1.
            };
            shape->SetTransMatrix( array );
#endif

            shape->SetPickable(false);
            // float x = (hitmap[it->first]->energy()*it->second)/maxEnergy;
            // x += 0.5*(1.0f - x*x*(3.0f - 2.0f*x));
            // shape->SetMainColorRGB( x, fmax(0.0f, 0.5f-x), 0.0f );
            // product->AddElement( shape ); 
            setupAddElement( shape, product );
         }
      }
   }
}

REGISTER_FWPROXYBUILDER( FWCaloParticleProxyBuilder, CaloParticleCollection, "CaloFTW", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );

#else
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
   const edm::EventBase *event = item()->getEvent();
   // hitmap
   std::map<DetId, const HGCRecHit*> hitmap;
   // max detected energy
   float maxEnergy(1e-5f);

   edm::Handle<HGCRecHitCollection> recHitHandleEE;
   edm::Handle<HGCRecHitCollection> recHitHandleFH;
   edm::Handle<HGCRecHitCollection> recHitHandleBH;   

   event->getByLabel( edm::InputTag( "HGCalRecHit", "HGCEERecHits" ), recHitHandleEE );
   event->getByLabel( edm::InputTag( "HGCalRecHit", "HGCHEFRecHits" ), recHitHandleFH );
   event->getByLabel( edm::InputTag( "HGCalRecHit", "HGCHEBRecHits" ), recHitHandleBH );

   const auto& rechitsEE = *recHitHandleEE;
   const auto& rechitsFH = *recHitHandleFH;
   const auto& rechitsBH = *recHitHandleBH;

   for (unsigned int i = 0; i < rechitsEE.size(); ++i) {
      hitmap[rechitsEE[i].detid().rawId()] = &rechitsEE[i];
      maxEnergy = fmax(maxEnergy, rechitsEE[i].energy());
   }
   for (unsigned int i = 0; i < rechitsFH.size(); ++i) {
      hitmap[rechitsFH[i].detid().rawId()] = &rechitsFH[i];
      maxEnergy = fmax(maxEnergy, rechitsFH[i].energy());   
   }
   for (unsigned int i = 0; i < rechitsBH.size(); ++i) {
      hitmap[rechitsBH[i].detid().rawId()] = &rechitsBH[i];
      maxEnergy = fmax(maxEnergy, rechitsBH[i].energy());
   }
   
   for (const auto & c : iData.simClusters())
   {
      auto clusterDetIds = (*c).hits_and_fractions();

      for( auto it = clusterDetIds.begin(), itEnd = clusterDetIds.end();
         it != itEnd; ++it )
      {

#if 1
         TEveGeoShape* wafer = item()->getGeom()->getHGCalEveShape(it->first);
         // wafer->SetDrawFrame(true);
         // wafer->SetHighlightFrame(true);
         // wafer->SetMiniFrame(true);
 
         // wafer->SetFillColor(0);
         // wafer->SetLineColor(100);
         // wafer->SetLineWidth(10);

         wafer->SetMainAlpha(0.5f); 
         // setupAddElement(wafer, &oItemHolder);
         oItemHolder.AddElement(wafer);

         TEveGeoShape* shape = new TEveGeoShape(TString::Format("CaloParticleRecHit Id=%u", it->first));

         const float* corners = item()->getGeom()->getCorners( it->first );
         if( corners == nullptr ) {
            continue;
         }

         // std::cout << (corners[6] + corners[0])*0.5f << "," << (corners[4] + corners[1])*0.5f << std::endl;
         float dx = abs(corners[6]  - corners[0]);
         float dy = abs(corners[4]  - corners[1]);
         float dz = abs(corners[14] - corners[2]);

         TGeoShape* geoShape = new TGeoBBox( dx, dy, dz );

         shape->SetShape( geoShape );
         double array[16] = { 1., 0., 0., 0.,
               0., 1., 0., 0.,
               0., 0., 1., 0.,
               (corners[6] + corners[0])*0.5f, (corners[4] + corners[1])*0.5f, (corners[14] + corners[2])*0.5f, 1.
         };
         shape->SetTransMatrix(  array );
         shape->SetMainAlpha(1.0f);
         // shape->SetMainTransparency(100); 
         // shape->SetMainColorRGB( 0.0f, 0.0f, 1.0f );
         oItemHolder.AddElement(shape);
#endif

         // setupAddElement(shape, &oItemHolder);
      }
   }
}

REGISTER_FWPROXYBUILDER( FWCaloParticleProxyBuilder, CaloParticle, "CaloFTWW", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
#endif
