#include "TEveCaloData.h"
#include "TEveViewer.h"
#include "TEveScene.h"
#include "TEveManager.h"
#include "TEveCalo.h"
#include "TColor.h"
#include "TAxis.h"
#include "TGLViewer.h"
#include "THLimitsFinder.h"
#include "TEveCaloLegoOverlay.h"

#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/FWLite/interface/Event.h"

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include <utility>


#include "TGeoMatrix.h"
#include "TEveTrans.h"

#include <utility>

TEveCaloLego* FWECALDetailViewBuilder::build()
{
   // get the hits from the event

   fwlite::Handle<EcalRecHitCollection> handle_hits;
   const EcalRecHitCollection *hits = 0;

   if (fabs(m_eta) < 1.5) {
      try {
         handle_hits.getByLabel(*m_event, "ecalRecHit", "EcalRecHitsEB");
         hits = handle_hits.ptr();
      }
      catch (...)
      {
         std::cout <<"no barrel ECAL rechits are available, "
            "showing crystal location but not energy" << std::endl;
      }
   } else {
      try {
         handle_hits.getByLabel(*m_event, "ecalRecHit", "EcalRecHitsEE");
         hits = handle_hits.ptr();
      }
      catch (...)
      {
         std::cout <<"no endcap ECAL rechits are available, "
            "showing crystal location but not energy" << std::endl;
      }
   }

   // data
   TEveCaloDataVec* data = new TEveCaloDataVec(1 + m_colors.size());
   data->IncDenyDestroy();
   data->RefSliceInfo(0).Setup("hits (not clustered)", 0.0, kMagenta+2);
   for (size_t i = 0; i < m_colors.size(); ++i)
   {
      data->RefSliceInfo(i + 1).Setup("hits (not clustered)", 0.0, m_colors[i]);
   }

   // fill
   fillData(hits, data);

   // axis
   Double_t etaMin(0), etaMax(0), phiMin(0), phiMax(0);
   data->GetEtaLimits(etaMin, etaMax);
   data->GetPhiLimits(phiMin, phiMax);
   Double_t bl, bh, bw;
   Int_t    bn, n = 20;
   THLimitsFinder::Optimize(etaMin, etaMax, n, bl, bh, bn, bw);
   data->SetEtaBins( new TAxis(bn, bl, bh));
   THLimitsFinder::Optimize(phiMin, phiMax, n, bl, bh, bn, bw);
   data->SetPhiBins( new TAxis(bn, bl, bh));
   if (fabs(m_eta) > 1.5) {
      data->GetEtaBins()->SetTitle("X[cm]");
      data->GetPhiBins()->SetTitle("Y[cm]");
   } else {
      data->GetEtaBins()->SetTitleFont(122);
      data->GetEtaBins()->SetTitle("h");
      data->GetPhiBins()->SetTitleFont(122);
      data->GetPhiBins()->SetTitle("f");
   }
   data->GetPhiBins()->SetTitleSize(0.03);
   data->GetEtaBins()->SetTitleSize(0.03);

   // lego
   TEveCaloLego *lego = new TEveCaloLego(data);
   lego->SetDrawNumberCellPixels(20);
   // scale and translate to real world coordinates
   lego->SetEta(etaMin, etaMax);
   lego->SetPhiWithRng((phiMin+phiMax)*0.5, (phiMax-phiMin)*0.5); // phi range = 2* phiOffset
   Double_t legoScale = ((etaMax - etaMin) < (phiMax - phiMin)) ? (etaMax - etaMin) : (phiMax - phiMin);
   lego->InitMainTrans();
   lego->RefMainTrans().SetScale(legoScale, legoScale, legoScale*0.5);
   lego->RefMainTrans().SetPos((etaMax+etaMin)*0.5, (phiMax+phiMin)*0.5, 0);
   lego->SetAutoRebin(kFALSE);
   lego->Set2DMode(TEveCaloLego::kValSize);
   lego->SetName("ECALDetail Lego");
   return lego;

}

void FWECALDetailViewBuilder::setColor(Color_t color, const std::vector<DetId> &detIds)
{

   m_colors.push_back(color);

   // get the slice for this group of detIds
   // note that the zeroth slice is the default one (all else)
   int slice = m_colors.size();
   // take a note of which slice these detids are going to go into
   for (size_t i = 0; i < detIds.size(); ++i)
      m_detIdsToColor[detIds[i]] = slice;
}

void FWECALDetailViewBuilder::showSuperCluster(const reco::SuperCluster &cluster, Color_t color)
{

   std::vector<DetId> clusterDetIds;
   const std::vector<std::pair<DetId, float> > &hitsAndFractions = cluster.hitsAndFractions();
   for (size_t j = 0; j < hitsAndFractions.size(); ++j)
   {
      clusterDetIds.push_back(hitsAndFractions[j].first);
   }

   setColor(color, clusterDetIds);

}

void FWECALDetailViewBuilder::showSuperClusters(Color_t color1, Color_t color2)
{

   // get the superclusters from the event

   fwlite::Handle<reco::SuperClusterCollection> handle_superclusters;
   const reco::SuperClusterCollection *superclusters = 0;

   if (fabs(m_eta) < 1.5) {
      try {
         handle_superclusters.getByLabel(*m_event, "correctedHybridSuperClusters");
         superclusters = handle_superclusters.ptr();
      }
      catch (...)
      {
         std::cout <<"no barrel superclusters are available" << std::endl;
      }
   } else {
      try {
         handle_superclusters.getByLabel(*m_event, "correctedMulti5x5SuperClustersWithPreshower");
         superclusters = handle_superclusters.ptr();
      }
      catch (...)
      {
         std::cout <<"no endcap superclusters are available" << std::endl;
      }
   }

   unsigned int colorIndex = 0;
   // sort clusters in eta so neighboring clusters have distinct colors
   reco::SuperClusterCollection sorted = *superclusters;
   std::sort(sorted.begin(), sorted.end(), superClusterEtaLess);
   for (size_t i = 0; i < sorted.size(); ++i)
   {
      if (!(fabs(sorted[i].eta() - m_eta) < (m_size*0.0172)
            && fabs(sorted[i].phi() - m_phi) < (m_size*0.0172)) )
         continue;

      if (colorIndex %2 == 0) showSuperCluster(sorted[i], color1);
      else showSuperCluster(sorted[i], color2);
      ++colorIndex;

   }

}

void FWECALDetailViewBuilder::fillData(const EcalRecHitCollection *hits,
                                       TEveCaloDataVec *data)
{

   // loop on all the detids
   for (EcalRecHitCollection::const_iterator k = hits->begin();
        k != hits->end(); ++k) {

      const TGeoHMatrix *matrix = m_geom->getMatrix(k->id().rawId());
      if ( matrix == 0 ) {
         printf("Warning: cannot get geometry for DetId: %d. Ignored.\n",k->id().rawId());
         continue;
      }

      TVector3 v(matrix->GetTranslation()[0],
                 matrix->GetTranslation()[1],
                 matrix->GetTranslation()[2]);

      // set the et
      double size = k->energy()/cosh(v.Eta());

      // check what slice to put in
      int slice = 0;
      std::map<DetId, int>::const_iterator itr = m_detIdsToColor.find(k->id());
      if (itr != m_detIdsToColor.end()) slice = itr->second;

      // if in the EB
      if (k->id().subdetId() == EcalBarrel) {

         // do phi wrapping
         double phi = v.Phi();
         if (v.Phi() > m_phi + M_PI)
            phi -= 2 * M_PI;
         if (v.Phi() < m_phi - M_PI)
            phi += 2 * M_PI;

         // check if the hit is in the window to be drawn
         if (!(fabs(v.Eta() - m_eta) < (m_size*0.0172)
               && fabs(phi - m_phi) < (m_size*0.0172)))
            continue;

         // if in the window to be drawn then draw it
         // data->AddTower(v.Eta() - 0.0172 / 2, v.Eta() + 0.0172 / 2,
         //                             phi - 0.0172 / 2, phi + 0.0172 / 2);
         DetIdToMatrix::Range range = m_geom->getEtaPhiRange(k->id().rawId());
         data->AddTower(range.min1, range.max1, range.min2, range.max2);
         data->FillSlice(slice, size);
         // if (size>0.5)
         // std::cout << k->id().rawId() << "\t Et:" << size << std::endl;

         // otherwise in the EE
      } else if (k->id().subdetId() == EcalEndcap) {

         // check if the hit is in the window to be drawn
         if (!(fabs(v.Eta() - m_eta) < (m_size*0.0172)
               && fabs(v.Phi() - m_phi) < (m_size*0.0172)))
            continue;

         // if in the window to be drawn then draw it
         DetIdToMatrix::Range range = m_geom->getXYRange(k->id().rawId());
         data->AddTower(range.min1, range.max1, range.min2, range.max2);
         // data->AddTower((v.X() - 2.9 / 2), (v.X() + 2.9 / 2),
         //             (v.Y() - 2.9 / 2), (v.Y() + 2.9 / 2));
         data->FillSlice(slice, size);
      }

   } // end loop on hits

   data->DataChanged();
}

