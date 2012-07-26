// FIXME - needed to set fixed eta-phi limits. Without the
//         visible area may change widely depending on energy
//         deposition availability

#include "TEveCaloData.h"
#include "TEveViewer.h"
#include "TEveCalo.h"
#include "TAxis.h"
#include "TMath.h"
#include "THLimitsFinder.h"
#include "TLatex.h"

#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "TGeoMatrix.h"
#include "TEveTrans.h"

#include <utility>

TEveCaloData* FWECALDetailViewBuilder::buildCaloData(bool xyEE)
{
   // get the hits from the event

   edm::Handle<EcalRecHitCollection> handle_hits;
   const EcalRecHitCollection *hits = 0;

   if (fabs(m_eta) < 1.5)
   {
      try
      {
         edm::InputTag tag("ecalRecHit", "EcalRecHitsEB");
         m_event->getByLabel(tag, handle_hits);
	 if (handle_hits.isValid())
         {
	    hits = &*handle_hits;
         }
      }
      catch (...)
      {
         fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::build():: Failed to access EcalRecHitsEB collection." << std::endl;
      }
      if ( ! handle_hits.isValid()) {
         try{
            edm::InputTag tag("reducedEcalRecHitsEB");
            m_event->getByLabel(tag, handle_hits);
            if (handle_hits.isValid())
            {
               hits = &*handle_hits;
            }

         }
         catch (...)
         {
            fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::build():: Failed to access reducedEcalRecHitsEB collection." << std::endl;
         }
      }   
   }
   else
   {
      try
      {
         edm::InputTag tag("ecalRecHit", "EcalRecHitsEE");
         m_event->getByLabel(tag, handle_hits);
	 if (handle_hits.isValid())
	    hits = &*handle_hits;
      }
      catch (...)
      {
         fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::build():: Failed to access ecalRecHitsEE collection." << std::endl;
      }

      if ( ! handle_hits.isValid()) {
         try {
            edm::InputTag tag("reducedEcalRecHitsEE");
            m_event->getByLabel(tag, handle_hits);
            if (handle_hits.isValid())
            {
               hits = &*handle_hits;
            }

         }
         catch (...)
         {     
            fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::build():: Failed to access reducedEcalRecHitsEE collection." << std::endl;
         }
      }
   }
     
   // data
   TEveCaloDataVec* data = new TEveCaloDataVec( 1 + m_colors.size() );
   data->RefSliceInfo(0).Setup("hits (not clustered)", 0.0, m_defaultColor );
   for( size_t i = 0; i < m_colors.size(); ++i )
   {
      data->RefSliceInfo(i + 1).Setup( "hits (clustered)", 0.0, m_colors[i] );
   }

   if( handle_hits.isValid() ) 
   {
      fillData( hits, data, xyEE );
   }

   // axis
   Double_t etaMin(0), etaMax(0), phiMin(0), phiMax(0);
   if (data->Empty())
   {
      fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::build():: No hits found in " << Form("FWECALDetailViewBuilder::build():: No hits found in eta[%f] phi[%f] region", m_eta, m_phi)<<".\n";

      // add dummy background
      float x = m_size*TMath::DegToRad();
      if (fabs(m_eta) < 1.5 || xyEE == false) {
         etaMin = m_eta -x;
         etaMax = m_eta +x;
         phiMin = m_phi -x;
         phiMax = m_phi +x;
         data->AddTower(etaMin, etaMax, phiMin, phiMax);
      }
      else
      {
         float theta = TEveCaloData::EtaToTheta(m_eta);
         float r = TMath::Tan(theta) * 290;
         phiMin = r * TMath::Cos(m_phi - x) -300;
         phiMax = r * TMath::Cos(m_phi + x) + 300;
         etaMin = r * TMath::Sin(m_phi - x) - 300;
         etaMax = r * TMath::Sin(m_phi + x) + 300;
         data->AddTower(TMath::Min(etaMin, etaMax), TMath::Max(etaMin, etaMax),
                        TMath::Min(phiMin, phiMax), TMath::Max(phiMin, phiMax));

      }
      data->FillSlice(0, 0.1);      
   }

 
   TAxis* eta_axis = 0;
   TAxis* phi_axis = 0; 
   data->GetEtaLimits(etaMin, etaMax);
   data->GetPhiLimits(phiMin, phiMax);
   //  printf("data rng %f %f %f %f\n",etaMin, etaMax, phiMin, phiMax );

   if (fabs(m_eta) > 1.5 && xyEE ) {
      eta_axis = new TAxis(10, etaMin, etaMax);
      phi_axis = new TAxis(10, phiMin, phiMax);
      eta_axis->SetTitle("X[cm]");
      phi_axis->SetTitle("Y[cm]");
      phi_axis->SetTitleSize(0.05);
      eta_axis->SetTitleSize(0.05);
   } else {
      std::vector<double> etaBinsWithinLimits;
      etaBinsWithinLimits.push_back(etaMin);
      for (unsigned int i=0; i<83; ++i)
         if ( fw3dlego::xbins[i] > etaMin && fw3dlego::xbins[i] < etaMax )
            etaBinsWithinLimits.push_back(fw3dlego::xbins[i]);
      etaBinsWithinLimits.push_back(etaMax);

      std::vector<double> phiBinsWithinLimits;
      phiBinsWithinLimits.push_back(phiMin);
      for ( double phi = -M_PI; phi < M_PI; phi += M_PI/36 )
         if ( phi > phiMin && phi < phiMax )
            phiBinsWithinLimits.push_back(phi);
      phiBinsWithinLimits.push_back(phiMax);

      eta_axis = new TAxis((int)etaBinsWithinLimits.size() -1, &etaBinsWithinLimits[0]);
      phi_axis = new TAxis((int)phiBinsWithinLimits.size() -1, &phiBinsWithinLimits[0]);

      eta_axis->SetTitleFont(122);
      eta_axis->SetTitle("h");
      eta_axis->SetTitleSize(0.07);
      phi_axis->SetTitleFont(122);
      phi_axis->SetTitle("f");
      phi_axis->SetTitleSize(0.07);
   }
   eta_axis->SetNdivisions(510);
   phi_axis->SetNdivisions(510);
   data->SetEtaBins(eta_axis);
   data->SetPhiBins(phi_axis);
   return data;
}

//_______________________________________________________________   
TEveCaloLego* FWECALDetailViewBuilder::build()
{
   TEveCaloData* data = buildCaloData(true);   
   
   double etaMin, etaMax, phiMin, phiMax;
   data->GetEtaLimits(etaMin, etaMax);
   data->GetPhiLimits(phiMin, phiMax);
   
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
   edm::Handle<reco::SuperClusterCollection> collection;

   if( fabs( m_eta ) < 1.5 ) {
      try {
         m_event->getByLabel(edm::InputTag("correctedHybridSuperClusters"), collection);
      }
      catch (...)
      {
         fwLog(fwlog::kWarning) <<"no barrel superclusters are available" << std::endl;
      }
   } else {
      try {
         m_event->getByLabel(edm::InputTag("correctedMulti5x5SuperClustersWithPreshower"), collection);
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
                                  TEveCaloDataVec *data, bool xyEE )
{
   const float barrelCR = m_size*0.0172; // barrel cell range
   
   // loop on all the detids
   for( EcalRecHitCollection::const_iterator k = hits->begin(), kEnd = hits->end();
       k != kEnd; ++k )
   {
      // get reco geometry
      double centerEta = 0;
      double centerPhi = 0;
      const float* points = m_geom->getCorners( k->id().rawId());
      if( points != 0 )
      {
         TEveVector v;
         int j = 0;
         for( int i = 0; i < 8; ++i )
         {	 
            v += TEveVector( points[j], points[j + 1], points[j + 2] );
            j +=3;
         }
         centerEta = v.Eta();
         centerPhi = v.Phi();
      }
      else
         fwLog( fwlog::kInfo ) << "cannot get geometry for DetId: "<< k->id().rawId() << ". Ignored.\n";
      
      double size = k->energy() / cosh( centerEta );
      
      // check what slice to put in
      int slice = 0;
      std::map<DetId, int>::const_iterator itr = m_detIdsToColor.find(k->id());
      if (itr != m_detIdsToColor.end()) slice = itr->second;
      
      // if in the EB
      if( k->id().subdetId() == EcalBarrel || xyEE == false )
      {
         // do phi wrapping
         if( centerPhi > m_phi + M_PI) centerPhi -= 2 * M_PI;
         if( centerPhi < m_phi - M_PI) centerPhi += 2 * M_PI;
         
         // check if the hit is in the window to be drawn
         if( !( fabs( centerEta - m_eta ) < barrelCR
               && fabs( centerPhi - m_phi ) < barrelCR )) continue;
         
         double minEta(10), maxEta(-10), minPhi(4), maxPhi(-4);
         if( points != 0 )
         {
            // calorimeter crystalls have slightly non-symetrical form in eta-phi projection
            // so if we simply get the largest eta and phi, cells will overlap
            // therefore we get a smaller eta-phi range representing the inner square
            // we also should use only points from the inner face of the crystal, since
            // non-projecting direction of crystals leads to large shift in eta on outter
            // face.
            int j = 0;
            float eps = 0.005;
            for( unsigned int i = 0; i < 8; ++i )
            {
               TEveVector crystal( points[j], points[j + 1], points[j + 2] );
               j += 3;
               double eta = crystal.Eta();
               double phi = crystal.Phi();
               if ( ((k->id().subdetId() == EcalBarrel)  && (crystal.Perp() > 135) )||  ((k->id().subdetId() == EcalEndcap) && (crystal.Perp() > 155))) continue;
               if ( minEta - eta > eps) minEta = eta;
               if ( eta - minEta > 0 && eta - minEta < eps ) minEta = eta;
               if ( eta - maxEta > eps) maxEta = eta;
               if ( maxEta - eta > 0 && maxEta - eta < eps ) maxEta = eta;
               if ( minPhi - phi > eps) minPhi = phi;
               if ( phi - minPhi > 0 && phi - minPhi < eps ) minPhi = phi;
               if ( phi - maxPhi > eps) maxPhi = phi;
               if ( maxPhi - phi > 0 && maxPhi - phi < eps ) maxPhi = phi;
            }
         }
         else 
         {
            double delta = 0.0172 * 0.5;
            minEta = centerEta - delta;
            maxEta = centerEta + delta;
            minPhi = centerPhi - delta;
            maxPhi = centerPhi + delta;
         }
         if( minPhi >= ( m_phi - barrelCR ) && maxPhi <= ( m_phi + barrelCR ) &&
            minEta >= ( m_eta - barrelCR ) && maxEta <= ( m_eta + barrelCR ))
         {
            // printf("add %f %f %f %f \n",minEta, maxEta, minPhi, maxPhi );
            data->AddTower( minEta, maxEta, minPhi, maxPhi );
            data->FillSlice( slice, size );
         }
         // otherwise in the EE
      }
      else if( k->id().subdetId() == EcalEndcap )
      {
         // check if the hit is in the window to be drawn
         double crystalSize = m_size * 0.0172;
         if( !( fabs( centerEta - m_eta ) < ( crystalSize )
               && fabs( centerPhi - m_phi ) < ( crystalSize )))
            continue;
         
         if( points != 0 )
         {
            double minX(9999), maxX(-9999), minY(9999), maxY(-9999);
            int j = 0;
            for( unsigned int i = 0; i < 8; ++i )
            {
               TEveVector crystal( points[j], points[j + 1], points[j + 2] );
               j += 3;
               double x = crystal.fX;
               double y = crystal.fY;
               if( fabs( crystal.fZ ) > 330 ) continue;
               if( minX - x > 0.01 ) minX = x;
               if( x - maxX > 0.01 ) maxX = x;
               if( minY - y > 0.01 ) minY = y;
               if( y - maxY > 0.01 ) maxY = y;
            }
            data->AddTower( minX, maxX, minY, maxY );
            // printf("EE add %f %f %f %f \n",minX, maxX, minY, maxY );
         }
         data->FillSlice( slice, size );
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
