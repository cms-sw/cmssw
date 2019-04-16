#ifndef Fireworks_Core_FWHeatmapProxyBuilderTemplate_h
#define Fireworks_Core_FWHeatmapProxyBuilderTemplate_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHeatmapProxyBuilderTemplate
//
/**\class FWHeatmapProxyBuilderTemplate FWHeatmapProxyBuilderTemplate.h Fireworks/Calo/interface/FWHeatmapProxyBuilderTemplate.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Alex Mourtziapis
//         Created:  Wed  Jan  23 14:50:00 EST 2019
//

// system include files
#include <cmath>

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilder.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

// forward declarations

template <typename T>
class FWHeatmapProxyBuilderTemplate : public FWSimpleProxyBuilder {

public:
   FWHeatmapProxyBuilderTemplate() :
      FWSimpleProxyBuilder(typeid(T)) {
   }

   //virtual ~FWHeatmapProxyBuilderTemplate();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

protected:
   std::map<DetId, const HGCRecHit*> hitmap;
   
   static constexpr uint8_t gradient_steps = 9;
   static constexpr uint8_t gradient[3][gradient_steps] = {
     {static_cast<uint8_t>(0.2082*255), static_cast<uint8_t>(0.0592*255), static_cast<uint8_t>(0.0780*255), 
      static_cast<uint8_t>(0.0232*255), static_cast<uint8_t>(0.1802*255), static_cast<uint8_t>(0.5301*255), 
      static_cast<uint8_t>(0.8186*255), static_cast<uint8_t>(0.9956*255), static_cast<uint8_t>(0.9764*255)},

     {static_cast<uint8_t>(0.1664*255), static_cast<uint8_t>(0.3599*255), static_cast<uint8_t>(0.5041*255), 
      static_cast<uint8_t>(0.6419*255), static_cast<uint8_t>(0.7178*255), static_cast<uint8_t>(0.7492*255), 
      static_cast<uint8_t>(0.7328*255), static_cast<uint8_t>(0.7862*255), static_cast<uint8_t>(0.9832*255)},

     {static_cast<uint8_t>(0.5293*255), static_cast<uint8_t>(0.8684*255), static_cast<uint8_t>(0.8385*255), 
      static_cast<uint8_t>(0.7914*255), static_cast<uint8_t>(0.6425*255), static_cast<uint8_t>(0.4662*255), 
      static_cast<uint8_t>(0.3499*255), static_cast<uint8_t>(0.1968*255), static_cast<uint8_t>(0.0539*255)}
   };

   const T& modelData(int index) { return *reinterpret_cast<const T*>(m_helper.offsetObject(item()->modelData(index))); }

   void setItem(const FWEventItem *iItem) override
   {
      FWProxyBuilderBase::setItem(iItem);
      if (iItem)
      {
         iItem->getConfig()->keepEntries(true);
         iItem->getConfig()->assertParam("Layer", 0L, 0L, 52L);
         iItem->getConfig()->assertParam("EnergyCutOff", 0.5, 0.2, 5.0);
         iItem->getConfig()->assertParam("Heatmap", true);
         iItem->getConfig()->assertParam("Z+", true);
         iItem->getConfig()->assertParam("Z-", true);
      }
   }

   void build(const FWEventItem *iItem, TEveElementList *product, const FWViewContext *vc) override
   {
      if (item()->getConfig()->template value<bool>("Heatmap"))
      {
         hitmap.clear();

         const edm::EventBase *event = iItem->getEvent();

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
         }
         for (unsigned int i = 0; i < rechitsFH.size(); ++i) {
            hitmap[rechitsFH[i].detid().rawId()] = &rechitsFH[i];
         }
         for (unsigned int i = 0; i < rechitsBH.size(); ++i) {
            hitmap[rechitsBH[i].detid().rawId()] = &rechitsBH[i];
         }
      }

      FWSimpleProxyBuilder::build(iItem, product, vc);
   }

   using FWSimpleProxyBuilder::build;
   void build(const void *iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* context) override
   {
      if(nullptr!=iData) {
         build(*reinterpret_cast<const T*> (iData), iIndex, oItemHolder, context);
      }
   }

   using FWSimpleProxyBuilder::buildViewType;
   void buildViewType(const void *iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType viewType, const FWViewContext* context) override
   {
      if(nullptr!=iData) {
         buildViewType(*reinterpret_cast<const T*> (iData), iIndex, oItemHolder, viewType, context);
      }
   }
   /**iIndex is the index where iData is found in the container from which it came
      iItemHolder is the object to which you add your own objects which inherit from TEveElement
   */
   virtual void build(const T& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*)
   {
      throw std::runtime_error("virtual build(const T&, unsigned int, TEveElement&, const FWViewContext*) not implemented by inherited class.");
   }

   virtual void buildViewType(const T& iData, unsigned int iIndex,TEveElement& oItemHolder, FWViewType::EType viewType, const FWViewContext*) 
   { 
      throw std::runtime_error("virtual buildViewType(const T&, unsigned int, TEveElement&, FWViewType::EType, const FWViewContext*) not implemented by inherited class");
   };
private:
   FWHeatmapProxyBuilderTemplate(const FWHeatmapProxyBuilderTemplate&) = delete; // stop default

   const FWHeatmapProxyBuilderTemplate& operator=(const FWHeatmapProxyBuilderTemplate&) = delete; // stop default



   // ---------- member data --------------------------------

};


#endif
