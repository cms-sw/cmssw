// FIXME - needed to set fixed eta-phi limits. Without the
//         visible area may change widely depending on energy
//         deposition availability

#include "TEveCaloData.h"
#include "TEveViewer.h"
#include "TEveCalo.h"
#include "TAxis.h"
#include "THLimitsFinder.h"
#include "TLatex.h"

#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

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
	 if( handle_hits.isValid() )
	    hits = handle_hits.ptr();
      }
      catch (...)
      {
         fwLog(fwlog::kWarning) <<"no barrel ECAL rechits are available, "
            "showing crystal location but not energy" << std::endl;
      }
   } else {
      try {
         handle_hits.getByLabel(*m_event, "ecalRecHit", "EcalRecHitsEE");
	 if( handle_hits.isValid() )
	    hits = handle_hits.ptr();
      }
      catch (...)
      {
         fwLog(fwlog::kWarning) <<"no endcap ECAL rechits are available, "
            "showing crystal location but not energy" << std::endl;
      }
   }
     
   // data
   TEveCaloDataVec* data = new TEveCaloDataVec( 1 + m_colors.size() );
   data->RefSliceInfo(0).Setup("hits (not clustered)", 0.0, m_defaultColor );
   for( size_t i = 0; i < m_colors.size(); ++i )
   {
      data->RefSliceInfo(i + 1).Setup( "hits (not clustered)", 0.0, m_colors[i] );
   }

   if( handle_hits.isValid() ) 
      // fill
      fillData( hits, data );

   // axis
   Double_t etaMin(0), etaMax(0), phiMin(0), phiMax(0);
   // it's hard to define properly visible area in X-Y,
   // so we rely on auto limits
   data->GetEtaLimits(etaMin, etaMax);
   data->GetPhiLimits(phiMin, phiMax);
   Double_t bl, bh, bw;
   Int_t bn, n = 20;
   THLimitsFinder::Optimize(etaMin, etaMax, n, bl, bh, bn, bw);
   data->SetEtaBins( new TAxis(bn, bl, bh));
   THLimitsFinder::Optimize(phiMin, phiMax, n, bl, bh, bn, bw);
   data->SetPhiBins( new TAxis(bn, bl, bh));

   // make tower grid
   std::vector<double> etaBinsWithinLimits;
   etaBinsWithinLimits.push_back(etaMin);
   for (unsigned int i=0; i<83; ++i)
      if ( fw3dlego::xbins[i] > etaMin && fw3dlego::xbins[i] < etaMax )
         etaBinsWithinLimits.push_back(fw3dlego::xbins[i]);
   etaBinsWithinLimits.push_back(etaMax);
   Double_t* eta_bins = new Double_t[etaBinsWithinLimits.size()];
   for (unsigned int i=0; i<etaBinsWithinLimits.size(); ++i)
      eta_bins[i] = etaBinsWithinLimits[i];

   std::vector<double> phiBinsWithinLimits;
   phiBinsWithinLimits.push_back(phiMin);
   for ( double phi = -M_PI; phi < M_PI; phi += M_PI/36 )
      if ( phi > phiMin && phi < phiMax ) // it's stupid, I know, but I'm lazy right now
         phiBinsWithinLimits.push_back(phi);
   phiBinsWithinLimits.push_back(phiMax);
   Double_t* phi_bins = new Double_t[phiBinsWithinLimits.size()];
   for (unsigned int i=0; i<phiBinsWithinLimits.size(); ++i)
      phi_bins[i] = phiBinsWithinLimits[i];
   if (fabs(m_eta) > 1.5) {
      data->GetEtaBins()->SetTitle("X[cm]");
      data->GetPhiBins()->SetTitle("Y[cm]");
      data->GetPhiBins()->SetTitleSize(0.03);
      data->GetEtaBins()->SetTitleSize(0.03);
   } else {
      data->SetEtaBins(new TAxis(etaBinsWithinLimits.size()-1,eta_bins));
      data->SetPhiBins(new TAxis(phiBinsWithinLimits.size()-1,phi_bins));
      data->GetEtaBins()->SetTitleFont(122);
      data->GetEtaBins()->SetTitle("h");
      data->GetPhiBins()->SetTitleFont(122);
      data->GetPhiBins()->SetTitle("f");
      data->GetPhiBins()->SetTitleSize(0.05);
      data->GetEtaBins()->SetTitleSize(0.05);
   }
   delete [] eta_bins;
   delete [] phi_bins;

   // lego
   TEveCaloLego *lego = new TEveCaloLego(data);
   lego->SetDrawNumberCellPixels(100);
   // scale and translate to real world coordinates
   lego->SetEta(etaMin, etaMax);
   lego->SetPhiWithRng((phiMin+phiMax)*0.5, (phiMax-phiMin)*0.5); // phi range = 2* phiOffset
   Double_t legoScale = ((etaMax - etaMin) < (phiMax - phiMin)) ? (etaMax - etaMin) : (phiMax - phiMin);
   lego->InitMainTrans();
   lego->RefMainTrans().SetScale(legoScale, legoScale, legoScale*0.5);
   lego->RefMainTrans().SetPos((etaMax+etaMin)*0.5, (phiMax+phiMin)*0.5, 0);
   lego->SetAutoRebin(kFALSE);
   lego->Set2DMode(TEveCaloLego::kValSizeOutline);
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

void
FWECALDetailViewBuilder::showSuperCluster( const reco::SuperCluster &cluster, Color_t color )
{
   std::vector<DetId> clusterDetIds;
   const std::vector<std::pair<DetId, float> > &hitsAndFractions = cluster.hitsAndFractions();
   for (size_t j = 0; j < hitsAndFractions.size(); ++j)
   {
      clusterDetIds.push_back(hitsAndFractions[j].first);
   }

   setColor( color, clusterDetIds );
}

void
FWECALDetailViewBuilder::showSuperClusters( Color_t color1, Color_t color2 )
{
   // get the superclusters from the event
   fwlite::Handle<reco::SuperClusterCollection> collection;

   if( fabs( m_eta ) < 1.5 ) {
      try {
         collection.getByLabel( *m_event, "correctedHybridSuperClusters" );
      }
      catch (...)
      {
         fwLog(fwlog::kWarning) <<"no barrel superclusters are available" << std::endl;
      }
   } else {
      try {
         collection.getByLabel( *m_event, "correctedMulti5x5SuperClustersWithPreshower" );
      }
      catch (...)
      {
         fwLog(fwlog::kWarning) <<"no endcap superclusters are available" << std::endl;
      }
   }
   if( collection.isValid() )
   {
      unsigned int colorIndex = 0;
      // sort clusters in eta so neighboring clusters have distinct colors
      reco::SuperClusterCollection sorted = *collection.product();
      std::sort( sorted.begin(), sorted.end(), superClusterEtaLess );
      for( size_t i = 0; i < sorted.size(); ++i )
      {
	 if( !(fabs(sorted[i].eta() - m_eta) < (m_size*0.0172)
	       && fabs(sorted[i].phi() - m_phi) < (m_size*0.0172)) )
	   continue;

	 if( colorIndex %2 == 0 )
	    showSuperCluster( sorted[i], color1 );
	 else
	    showSuperCluster( sorted[i], color2 );
	 ++colorIndex;
      }
   }
}

void
FWECALDetailViewBuilder::fillData( const EcalRecHitCollection *hits,
				   TEveCaloDataVec *data )
{
   const float barrelCR = m_size*0.0172; // barrel cell range

   // loop on all the detids
   for( EcalRecHitCollection::const_iterator k = hits->begin(), kEnd = hits->end();
        k != kEnd; ++k ) {
      const TGeoHMatrix *matrix = m_geom->getMatrix( k->id().rawId() );
      if( matrix == 0 ) {
         fwLog(fwlog::kInfo) << " cannot get geometry for DetId: "<< k->id().rawId()  <<". Ignored.\n";
         continue;
      }

      TVector3 v( matrix->GetTranslation()[0],
		  matrix->GetTranslation()[1],
		  matrix->GetTranslation()[2] );

      // set the et
      double size = k->energy()/cosh(v.Eta());

      // check what slice to put in
      int slice = 0;
      std::map<DetId, int>::const_iterator itr = m_detIdsToColor.find(k->id());
      if (itr != m_detIdsToColor.end()) slice = itr->second;

      // get reco geometry
      const std::vector<TEveVector>& points = m_geom->getPoints(k->id().rawId());

      // if in the EB
      if (k->id().subdetId() == EcalBarrel) {
         // do phi wrapping
         double phi = v.Phi();
         if (v.Phi() > m_phi + M_PI) phi -= 2 * M_PI;
         if (v.Phi() < m_phi - M_PI) phi += 2 * M_PI;

         // check if the hit is in the window to be drawn
         if (!(fabs(v.Eta() - m_eta) < barrelCR
               && fabs(phi - m_phi) < barrelCR)) continue;

         double minEta(10), maxEta(-10), minPhi(4), maxPhi(-4);
         if ( points.size() == 8 ) {
            // calorimeter crystalls have slightly non-symetrical form in eta-phi projection
            // so if we simply get the largest eta and phi, cells will overlap
            // therefore we get a smaller eta-phi range representing the inner square
            // we also should use only points from the inner face of the crystal, since
            // non-projecting direction of crystals leads to large shift in eta on outter
            // face.
            for (unsigned int i=0; i<points.size(); ++i)
            {
               double eta = points[i].Eta();
               double phi = points[i].Phi();
               if ( points[i].Perp() > 135 ) continue;
               if ( minEta - eta > 0.01) minEta = eta;
               if ( eta - minEta > 0 && eta - minEta < 0.01 ) minEta = eta;
               if ( eta - maxEta > 0.01) maxEta = eta;
               if ( maxEta - eta > 0 && maxEta - eta < 0.01 ) maxEta = eta;
               if ( minPhi - phi > 0.01) minPhi = phi;
               if ( phi - minPhi > 0 && phi - minPhi < 0.01 ) minPhi = phi;
               if ( phi - maxPhi > 0.01) maxPhi = phi;
               if ( maxPhi - phi > 0 && maxPhi - phi < 0.01 ) maxPhi = phi;
            }
         }
         else 
         {
            minEta = v.Eta() - 0.0172 / 2;
            maxEta = v.Eta() + 0.0172 / 2;
            minPhi = phi     - 0.0172 / 2;
            maxPhi = phi     + 0.0172 / 2;
         }
         if (minPhi >= (m_phi-barrelCR) && maxPhi <= (m_phi+barrelCR) &&
             minEta >= (m_eta-barrelCR) && maxEta <= (m_eta+barrelCR))
         {
            data->AddTower(minEta, maxEta, minPhi, maxPhi);
            data->FillSlice(slice, size);
         }
         // otherwise in the EE
      } else if (k->id().subdetId() == EcalEndcap) {

         // check if the hit is in the window to be drawn
         if (!(fabs(v.Eta() - m_eta) < (m_size*0.0172)
               && fabs(v.Phi() - m_phi) < (m_size*0.0172)))
            continue;

         if ( points.size() == 8 ) {
            double minX(9999), maxX(-9999), minY(9999), maxY(-9999);
            for (unsigned int i=0; i<points.size(); ++i) {
               double x = points[i].fX;
               double y = points[i].fY;
               if ( fabs(points[i].fZ) > 330 ) continue;
               if ( minX - x > 0.01) minX = x;
               if ( x - maxX > 0.01) maxX = x;
               if ( minY - y > 0.01) minY = y;
               if ( y - maxY > 0.01) maxY = y;
            }
            data->AddTower(minX, maxX, minY, maxY);
         } else {
            data->AddTower((v.X() - 2.9 / 2), (v.X() + 2.9 / 2),
                           (v.Y() - 2.9 / 2), (v.Y() + 2.9 / 2));
         }
         data->FillSlice(slice, size);
      }
   } // end loop on hits

   data->DataChanged();
}

double
FWECALDetailViewBuilder::makeLegend( double x0, double y0,
                                     Color_t clustered1, Color_t clustered2,
                                     Color_t supercluster
                                     )
{
   Double_t fontsize = 0.07;
   TLatex* latex = new TLatex();
   Double_t x = x0;
   Double_t y = y0;
   Double_t boxH = 0.25*fontsize;
   Double_t yStep = 0.04;

   y -= yStep;
   latex->DrawLatex(x, y, "Energy types:");
   y -= yStep;

   Double_t pos[4];
   pos[0] = x+0.05;
   pos[2] = x+0.20;

   pos[1] = y; pos[3] = pos[1] + boxH;
   FWDetailViewBase::drawCanvasBox(pos, m_defaultColor);
   latex->DrawLatex(x+0.25, y, "unclustered");
   y -= yStep;
   if (clustered1<0) return y;

   pos[1] = y; pos[3] = pos[1] + boxH;
   FWDetailViewBase::drawCanvasBox(pos, clustered1);
   latex->DrawLatex(x+0.25, y, "clustered");
   y -= yStep;
   if (clustered2<0) return y;

   pos[1] = y; pos[3] = pos[1] + boxH;
   FWDetailViewBase::drawCanvasBox(pos, clustered2);
   latex->DrawLatex(x+0.25, y, "clustered");
   y -= yStep;
   if (supercluster<0) return y;

   pos[1] = y; pos[3] = pos[1] + boxH;
   FWDetailViewBase::drawCanvasBox(pos, supercluster);
   latex->DrawLatex(x+0.25, y, "super-cluster");
   y -= yStep;

   return y;
}
