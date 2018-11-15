// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloParticleTowerProxyBuilderBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 11:28:28 EST 2008
//

// system includes
#include <cmath>

// user includes
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "TH2F.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"

#include "Fireworks/Core/interface/fw3dlego_xbins.h"
#include "Fireworks/Calo/plugins/FWCaloParticleTowerProxyBuilder.h"
#include "Fireworks/Calo/plugins/FWCaloTowerSliceSelector.h"

#include "FWCore/Common/interface/EventBase.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"

//---- display CaloParticle Rechits ----
// #define SHOW_RECHITS

//
// constructors , dectructors
//
FWCaloParticleTowerProxyBuilderBase::FWCaloParticleTowerProxyBuilderBase() : FWCaloDataHistProxyBuilder(),
                                                                             m_towers(nullptr)
{
}

FWCaloParticleTowerProxyBuilderBase::~FWCaloParticleTowerProxyBuilderBase()
{
}

void FWCaloParticleTowerProxyBuilderBase::build(const FWEventItem *iItem,
                                                TEveElementList *el, const FWViewContext *ctx)
{
   m_towers = nullptr;
   if (iItem)
   {
      iItem->get(m_towers);
      FWCaloDataProxyBuilderBase::build(iItem, el, ctx);
   }
}

FWHistSliceSelector *
FWCaloParticleTowerProxyBuilderBase::instantiateSliceSelector()
{
   return new FWCaloTowerSliceSelector(m_hist, item());
}

void FWCaloParticleTowerProxyBuilderBase::fillCaloData()
{
   m_hist->Reset();

   if (!m_towers || !item()->defaultDisplayProperties().isVisible())
      return;

#ifdef SHOW_RECHITS
   using namespace TMath;
   const static float upPhiLimit = Pi() - 10 * DegToRad() - 1e-5;

   // hitmap
   std::map<DetId, const HGCRecHit *> hitmap;
   {
      edm::Handle<HGCRecHitCollection> recHitHandleEE;
      edm::Handle<HGCRecHitCollection> recHitHandleFH;
      edm::Handle<HGCRecHitCollection> recHitHandleBH;

      const edm::EventBase *event = item()->getEvent();
      event->getByLabel(edm::InputTag("HGCalRecHit", "HGCEERecHits"), recHitHandleEE);
      event->getByLabel(edm::InputTag("HGCalRecHit", "HGCHEFRecHits"), recHitHandleFH);
      event->getByLabel(edm::InputTag("HGCalRecHit", "HGCHEBRecHits"), recHitHandleBH);

      const auto &rechitsEE = *recHitHandleEE;
      const auto &rechitsFH = *recHitHandleFH;
      const auto &rechitsBH = *recHitHandleBH;

      for (unsigned int i = 0; i < rechitsEE.size(); ++i)
      {
         hitmap[rechitsEE[i].detid().rawId()] = &rechitsEE[i];
      }
      for (unsigned int i = 0; i < rechitsFH.size(); ++i)
      {
         hitmap[rechitsFH[i].detid().rawId()] = &rechitsFH[i];
      }
      for (unsigned int i = 0; i < rechitsBH.size(); ++i)
      {
         hitmap[rechitsBH[i].detid().rawId()] = &rechitsBH[i];
      }
   }
#endif

   unsigned int index = 0;
   for (CaloParticleCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower, ++index)
   {
      const FWEventItem::ModelInfo &info = item()->modelInfo(index);
      if (!info.displayProperties().isVisible())
         continue;

#ifdef SHOW_RECHITS
      for (const auto &c : tower->simClusters())
      {
         for (const auto &it : (*c).hits_and_fractions())
         {
            if (hitmap.find(it.first) == hitmap.end())
               continue;

            const float *corners = item()->getGeom()->getCorners(it.first);

            if (corners == nullptr)
               continue;

            std::vector<TEveVector> front(6);
            float eta[6], phi[6];
            bool plusSignPhi = false;
            bool minusSignPhi = false;
            int j = 0;
            for (int i = 0; i < 6; ++i)
            {
               front[i] = TEveVector(corners[j], corners[j + 1], corners[j + 2]);
               j += 3;

               eta[i] = front[i].Eta();
               phi[i] = front[i].Phi();

               // make sure sign around Pi is same as sign of fY
               phi[i] = Sign(phi[i], front[i].fY);

               (phi[i] >= 0) ? plusSignPhi = true : minusSignPhi = true;
            }

            // check for cell around phi and move up edge to negative side
            if (plusSignPhi && minusSignPhi)
            {
               for (int i = 0; i < 6; ++i)
               {
                  if (phi[i] >= upPhiLimit)
                  {
                     //  printf("over phi max limit %f \n", phi[i]);
                     phi[i] -= TwoPi();
                  }
               }
            }

            float etaM = -10;
            float etam = 10;
            float phiM = -6;
            float phim = 6;
            for (int i = 0; i < 6; ++i)
            {
               etam = Min(etam, eta[i]);
               etaM = Max(etaM, eta[i]);
               phim = Min(phim, phi[i]);
               phiM = Max(phiM, phi[i]);
            }

            addEntryToTEveCaloData((etam + etaM) * 0.5, (phim + phiM) * 0.5, hitmap[it.first]->energy() * it.second, info.isSelected());
         }
      }
#else
      addEntryToTEveCaloData(tower->eta(), tower->phi(), tower->et(), info.isSelected());
#endif
   }
}

REGISTER_FWPROXYBUILDER(FWCaloParticleTowerProxyBuilderBase, CaloParticleCollection, "CaloPLeg000ooo", FWViewType::k3DBit | FWViewType::kAllRPZBits | FWViewType::kAllLegoBits);
