// FIXME - needed to set fixed eta-phi limits. Without the
//         visible area may change widely depending on energy
//         deposition availability

#include "TEveCaloData.h"
#include "TEveViewer.h"
#include "TEvePointSet.h"
#include "TEveCalo.h"
#include "TEveCompound.h"
#include "TAxis.h"
#include "TMath.h"
#include "THLimitsFinder.h"
#include "TLatex.h"

#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Calo/interface/FWBoxRecHit.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/Context.h"


#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "TGeoMatrix.h"
#include "TEveTrans.h"

#include <utility>


FWECALDetailViewBuilder::FWECALDetailViewBuilder(const edm::EventBase *event, const FWGeometry* geom, 
                                                 float eta, float phi, int size , Color_t defaultColor)
                                                
: m_event(event), m_geom(geom),
  m_eta(eta), m_phi(phi), m_size(size),
  m_defaultColor(defaultColor), m_towerList(nullptr)
{
}


TEveCaloData* FWECALDetailViewBuilder::buildCaloData(bool)
{
   // get the hits from the event

   // data
   TEveCaloDataVec* data = new TEveCaloDataVec( 1);
   data->SetWrapTwoPi(false);
   data->RefSliceInfo(0).Setup("hits (not clustered)", 0.0, m_defaultColor );
   
   fillData(data);

   // axis
   float etaMin = m_eta - sizeRad();
   float etaMax = m_eta + sizeRad();
   float phiMin = m_phi - sizeRad();
   float phiMax = m_phi + sizeRad();

   data->AddTower(m_eta - sizeRad(), m_eta + sizeRad(), m_phi - sizeRad(), m_phi + sizeRad());

   data->FillSlice(0, 0.1);      
   

 
   TAxis* eta_axis = nullptr;
   TAxis* phi_axis = nullptr; 

   //  printf("data rng %f %f %f %f\n",etaMin, etaMax, phiMin, phiMax );  
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

   eta_axis->SetNdivisions(510);
   phi_axis->SetNdivisions(510);
   data->SetEtaBins(eta_axis);
   data->SetPhiBins(phi_axis);
   return data;
}

//_______________________________________________________________   
TEveCaloLego* FWECALDetailViewBuilder::build()
{

   // axis
   float etaMin = m_eta - sizeRad();
   float etaMax = m_eta + sizeRad();
   float phiMin = m_phi - sizeRad();
   float phiMax = m_phi + sizeRad();
   
   m_towerList = new TEveElementList("TowerHolder");
   TEveCaloData* data = buildCaloData(true);
      
   // lego
   TEveCaloLego *lego = new TEveCaloLego();
   lego->SetData(data);
   lego->AddElement(m_towerList);
   lego->SetAutoRange(false);
   lego->SetDrawNumberCellPixels(100);
   // scale and translate to real world coordinates
   lego->SetEta(etaMin, etaMax);
   lego->SetPhiWithRng((phiMin+phiMax)*0.5, (phiMax-phiMin)*0.5); // phi range = 2* phiOffset
   Double_t legoScale = sizeRad() *2;
   lego->InitMainTrans();
   lego->RefMainTrans().SetScale(legoScale, legoScale, legoScale*0.5);
   lego->RefMainTrans().SetPos(m_eta, m_phi, -0.01);
   lego->SetAutoRebin(kFALSE);
   lego->SetName("ECALDetail Lego");

   // cut & paste from FWLegoViewBase
   lego->SetScaleAbs(true);
   lego->SetHasFixedHeightIn2DMode(true);
   lego->SetFixedHeightValIn2DMode(0.001);

   
   TEvePointSet* ps = new TEvePointSet("origin");
   ps->SetNextPoint(m_eta, m_phi, 0.01);
   ps->SetMarkerSize(0.05);
   ps->SetMarkerStyle(2);
   ps->SetMainColor(kGreen);
   ps->SetMarkerColor(kGreen);
   lego->AddElement(ps);

   return lego;

}

void FWECALDetailViewBuilder::setColor(Color_t color, const std::vector<DetId> &detIds)
{
   for (size_t i = 0; i < detIds.size(); ++i)
      m_detIdsToColor[detIds[i]] = color;
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
	 if( !(fabs(sorted[i].eta() - m_eta) < sizeRad()
	       && fabs(sorted[i].phi() - m_phi) < sizeRad()) )
	   continue;

	 if( colorIndex %2 == 0 )
	    showSuperCluster( sorted[i], color1 );
	 else
	    showSuperCluster( sorted[i], color2 );
	 ++colorIndex;
      }
   }
}


namespace {
float
calculateEt( const TEveVector &centre, float e )
{
   TEveVector vec = centre;
   float et;

   vec.Normalize();
   vec *= e;
   et = vec.Perp();

   return et;
}

}
//------------------------------------------------------------------
void
FWECALDetailViewBuilder::fillEtaPhi( const EcalRecHitCollection *hits,TEveCaloDataVec *data)
{
    // printf("filletaphi \n");  
   const float area = sizeRad(); // barrel cell range, AMT this is available in context

   double eta1 = m_eta - area;
   double eta2 = m_eta + area;
   double phi1 = m_phi - area;
   double phi2 = m_phi + area;


   std::vector<FWBoxRecHit*>  boxes;
   for( EcalRecHitCollection::const_iterator hitIt = hits->begin(); hitIt != hits->end(); ++hitIt)
   {    
       const float *corners = m_geom->getCorners( hitIt->detid() );
       float energy, et;
       std::vector<TEveVector> etaphiCorners(8);

       if( corners == nullptr )
           continue;


       for( int i = 0; i < 4; ++i )
       {
           TEveVector cv = TEveVector( corners[i*3], corners[i*3+1], corners[i*3+2] );
           etaphiCorners[i].fX = cv.Eta();                                     // Conversion of rechit X/Y values for plotting in Eta/Phi
           etaphiCorners[i].fY = cv.Phi();
           etaphiCorners[i].fZ = 0.0;

           etaphiCorners[i+4].fX = etaphiCorners[i].fX;                        // Top can simply be plotted exactly over the top of the bottom face
           etaphiCorners[i+4].fY = etaphiCorners[i].fY;
           etaphiCorners[i+4].fZ = 0.001;
           //  printf("%f %f %d \n",  etaphiCorners[i].fX, etaphiCorners[i].fY, i);
       }

       TEveVector center;
       for( int i = 0; i < 4; ++i )
           center += etaphiCorners[i];
       center *= 1.f / 4.f;

       
       if ( center.fX < eta1 || center.fX > eta2) continue;
       if ( center.fY < phi1 || center.fY > phi2) continue;



       // Stop phi wrap
       float dPhi1 = etaphiCorners[2].fY - etaphiCorners[1].fY;
       float dPhi2 = etaphiCorners[3].fY - etaphiCorners[0].fY;
       float dPhi3 = etaphiCorners[1].fY - etaphiCorners[2].fY;
       float dPhi4 = etaphiCorners[0].fY - etaphiCorners[3].fY;

       if( dPhi1 > 1 )
           etaphiCorners[2].fY = etaphiCorners[2].fY - ( 2 * TMath::Pi() );
       if( dPhi2 > 1 )
           etaphiCorners[3].fY = etaphiCorners[3].fY - ( 2 * TMath::Pi() );
       if( dPhi3 > 1 )
           etaphiCorners[2].fY = etaphiCorners[2].fY + ( 2 * TMath::Pi() );
       if( dPhi4 > 1 )
           etaphiCorners[3].fY = etaphiCorners[3].fY + ( 2 * TMath::Pi() );



       energy = hitIt->energy();
       et = calculateEt( center, energy );
       Color_t bcolor = m_defaultColor;
       std::map<DetId, int>::const_iterator itr = m_detIdsToColor.find(hitIt->id());
       if (itr != m_detIdsToColor.end()) bcolor = itr->second;

       m_boxes.push_back(new FWBoxRecHit( etaphiCorners, m_towerList, energy, et ));
       TEveElement::List_i pIt = m_boxes.back()->getTower()->BeginParents();
       TEveCompound* comp = dynamic_cast<TEveCompound*>(*pIt);
       comp->SetMainColor(bcolor);
       m_boxes.back()->getTower()->SetPickable(true);
       m_boxes.back()->getTower()->SetElementTitle(Form("rawId = %d, et = %f", hitIt->id().rawId(), et));
   } // loop hits

}


//---------------------------------------------------------------------------------------


void
FWECALDetailViewBuilder::fillData(  TEveCaloDataVec *data)
{
   { // barrel
      const EcalRecHitCollection *hitsEB = nullptr;
      edm::Handle<EcalRecHitCollection> handle_hitsEB;

      // RECO
      try
      {
         edm::InputTag tag("ecalRecHit", "EcalRecHitsEB");
         m_event->getByLabel(tag, handle_hitsEB);
	 if (handle_hitsEB.isValid())
         {
	    hitsEB = &*handle_hitsEB;
         }
      }
      catch (...)
      {
         fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::fillData():: Failed to access EcalRecHitsEB collection." << std::endl;
      }


      // AOD
      if ( ! handle_hitsEB.isValid()) {
         try{
            edm::InputTag tag("reducedEcalRecHitsEB");
            m_event->getByLabel(tag, handle_hitsEB);
            if (handle_hitsEB.isValid())
            {
               hitsEB = &*handle_hitsEB;
            }

         }
         catch (...)
         {
            fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::filData():: Failed to access reducedEcalRecHitsEB collection." << std::endl;
         }
      }

      // MINIAOD
      if ( ! handle_hitsEB.isValid()) {
         try{
            edm::InputTag tag("reducedEgamma", "reducedEBRecHits");
            m_event->getByLabel(tag, handle_hitsEB);
            if (handle_hitsEB.isValid())
            {
               hitsEB = &*handle_hitsEB;
            }




         }
         catch (...)
         {
            fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::filData():: Failed to access reducedEgamma collection." << std::endl;
         }
      }

      if( handle_hitsEB.isValid() ) 
      {
         fillEtaPhi( hitsEB, data);
      }
   }

   {// endcap

      const EcalRecHitCollection *hitsEE = nullptr;
      edm::Handle<EcalRecHitCollection> handle_hitsEE;

      // RECO
      try
      {
         edm::InputTag tag("ecalRecHit", "EcalRecHitsEE");
         m_event->getByLabel(tag, handle_hitsEE);
	 if (handle_hitsEE.isValid())
	    hitsEE = &*handle_hitsEE;
      }
      catch (...)
      {
         fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::fillData():: Failed to access ecalRecHitsEE collection." << std::endl;
      }

      // AOD
      if ( ! handle_hitsEE.isValid()) {
         try {
            edm::InputTag tag("reducedEcalRecHitsEE");
            m_event->getByLabel(tag, handle_hitsEE);
            if (handle_hitsEE.isValid())
            {
               hitsEE = &*handle_hitsEE;
            }

         }
         catch (...)
         {     
            fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::fillData():: Failed to access reducedEcalRecHitsEE collection." << std::endl;
         }

      // MINIAOD
      if ( ! handle_hitsEE.isValid()) {
         try {
            edm::InputTag tag("reducedEgamma", "reducedEERecHits");
            m_event->getByLabel(tag, handle_hitsEE);
            if (handle_hitsEE.isValid())
            {
               hitsEE = &*handle_hitsEE;
            }

         }
         catch (...)
         {     
            fwLog(fwlog::kWarning) <<"FWECALDetailViewBuilder::fillData():: Failed to access reducedEcalRecHitsEE collection." << std::endl;
         }
      }
   
      }

      if( handle_hitsEE.isValid() ) 
      {
          fillEtaPhi( hitsEE, data);
      }
   }

   if ( m_boxes.empty()) return;

   bool plotEt = true;
   float maxEnergy = 0;
   int maxEnergyIdx = 0;
   // get max energy in EE and EB

   int cnt = 0;
   for (auto & i : m_boxes)  {
       if (i->getEnergy(plotEt) > maxEnergy) {
           maxEnergy = i->getEnergy(plotEt);
           maxEnergyIdx = cnt;
       }
       cnt++;
   }

   m_boxes[maxEnergyIdx]->setIsTallest();

   // AMT ... max size can be an external parameter
   float scale = 0.3/maxEnergy;
   for (auto & i : m_boxes) {
        i->updateScale(scale, log(maxEnergy + 1), plotEt);
        i->getTower()->SetDrawFrame(true);
   }
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
//______________________________________________________________________________

float FWECALDetailViewBuilder::sizeRad() const
{
   float rs = m_size * TMath::DegToRad();
   return rs;
}

