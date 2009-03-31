// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWElectronDetailView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWElectronDetailView.cc,v 1.15 2009/03/29 14:13:38 amraktad Exp $
//

// system include files
#include "TLatex.h"
#include "TAxis.h"
#include "TGeoBBox.h"

#include "TGLViewer.h"

#include "TEveManager.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#define protected public
#include "TEveLegoEventHandler.h"
#undef protected
#include "TEveCaloLegoOverlay.h"
#include "TEveText.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

// user include files
#include "Fireworks/Electrons/plugins/FWElectronDetailView.h"

#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWElectronDetailView::FWElectronDetailView():
   m_item(0),
   m_coordEtaPhi(true),
   m_barrel_hits(0),
   m_endcap_hits(0),
   m_endcap_reduced_hits(0)
{

}

FWElectronDetailView::~FWElectronDetailView()
{
   resetCenter();
}

//
// member functions
//
TEveElement* FWElectronDetailView::build (const FWModelId &id, const reco::GsfElectron* iElectron)
{
   return build_projected(id, iElectron);
}

TEveElement* FWElectronDetailView::build_projected (const FWModelId &id,
                                                    const reco::GsfElectron* iElectron)
{
   // printf("calling FWElectronDetailView::buildRhoZ\n");
   if(0==iElectron) { return 0;}
   m_item = id.item();

   viewer()->SetCurrentCamera(TGLViewer::kCameraPerspXOY);

   TEveElementList* tList =  new TEveElementList(m_item->name().c_str(),"Supercluster RhoZ",true);
   tList->SetMainColor(m_item->defaultDisplayProperties().color());
   gEve->AddElement(tList);

   // get rechits
   const fwlite::Event *ev = m_item->getEvent();
   fwlite::Handle<EcalRecHitCollection> handle_barrel_hits;
   m_barrel_hits = 0;
   try {
      handle_barrel_hits.getByLabel(*ev, "caloRecHits", "EcalRecHitsEB");
      m_barrel_hits = handle_barrel_hits.ptr();
   }
   catch (...) 
   {
      std::cout <<"no barrel ECAL rechits are available, "
         "showing crystal location but not energy" << std::endl;
   }
   fwlite::Handle<EcalRecHitCollection> handle_endcap_hits;
   m_endcap_hits = 0;
   try {
      handle_endcap_hits.getByLabel(*ev, "caloRecHits", "EcalRecHitsEE");
      m_endcap_hits = handle_endcap_hits.ptr();
   }
   catch ( ...)
   {
      std::cout <<"no endcap ECAL rechits are available, "
         "showing crystal location but not energy" << std::endl;
   }
//    fwlite::Handle<EcalRecHitCollection> handle_endcap_reduced_hits;
//    m_endcap_reduced_hits = 0;
//    try {
//       handle_endcap_reduced_hits.getByLabel(*ev, "reducedEcalRecHitsEE");
//       m_endcap_reduced_hits = handle_endcap_reduced_hits.ptr();
//    }
//    catch ( ...)
//    {
//       std::cout <<"no endcap ECAL reduced rechits are available, "
//          "not showing surrounding hits" << std::endl;
//    }
   if (const reco::GsfElectron *i = iElectron) {
      m_coordEtaPhi = true;
      assert(i->gsfTrack().isNonnull());
      assert(i->superCluster().isNonnull());
      tList->AddElement(makeLabels(*i));
      std::vector<DetId> detids = i->superCluster()->getHitsByDetId();
      seed_detids = i->superCluster()->seed()->
         getHitsByDetId();
      const unsigned int subdetId = 
         seed_detids.size() != 0 ? seed_detids.begin()->subdetId() : 0;
      if (subdetId == EcalBarrel) {
         rotationCenter()[0] = i->superCluster()->position().eta();
         rotationCenter()[1] = i->superCluster()->position().phi();
         rotationCenter()[2] = 0;
      } else if (subdetId == EcalEndcap) {
         m_coordEtaPhi = false;
         rotationCenter()[0] = i->superCluster()->position().x()*0.01;
         rotationCenter()[1] = i->superCluster()->position().y()*0.01;
         rotationCenter()[2] = 0;
      }
      //        rotationCenter()[0] = i->TrackPositionAtCalo().x();
      //        rotationCenter()[1] = i->TrackPositionAtCalo().y();
      //        rotationCenter()[2] = i->TrackPositionAtCalo().z();
      // vector for the ECAL crystals
      TEveCaloDataVec* data = new TEveCaloDataVec(2);
      // one slice for the seed cluster (red) and one for the
      // other clusters
      data->RefSliceInfo(0).Setup("seed cluster", 0.0, kRed);
      data->RefSliceInfo(1).Setup("other clusters", 0.0, kYellow);
//       data->RefSliceInfo(2).Setup("unclustered crystals", 0.0, kYellow);
      // now fill
      fillData(detids, data, i->superCluster()->seed()->position().phi());
      if (m_endcap_reduced_hits != 0)
	   fillReducedData(detids, data);
      data->DataChanged();
      // printf("max val %f  %f \n", data->GetMaxVal(0), data->GetMaxVal(1));
         
      // lego
      TEveCaloLego* lego = new TEveCaloLego(data);
      // scale and translate to real world coordinates
      Double_t em, eM, pm, pM;
      data->GetEtaLimits(em, eM);
      data->GetPhiLimits(pm, pM);
      // printf("eta limits %f %f  phi linits %f %f \n", em, eM, pm, pM);
      lego->SetEta(em, eM);
      lego->SetPhiWithRng((pm+pM)*0.5, (pM-pm)*0.5); // phi range = 2* phiOffset
      Double_t legoScale = ((eM - em) < (pM - pm)) ? (eM - em) : (pM - pm);
      lego->InitMainTrans();
      lego->RefMainTrans().SetScale(legoScale, legoScale, legoScale);
      lego->RefMainTrans().SetPos((eM+em)*0.5, (pM+pm)*0.5, 0);

      lego->SetAutoRebin(kFALSE);
      lego->Set2DMode(TEveCaloLego::kValSize);
      lego->SetProjection(TEveCaloLego::kAuto);
      lego->SetName("ElectronDetail Lego");
      gEve->AddElement(lego, tList);

   
      // overlay lego  (only a debug info == axis at the corner)       
      TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
      overlay->SetShowPlane(kFALSE);
      overlay->SetShowPerspective(kFALSE);
      overlay->SetCaloLego(lego);
      viewer()->AddOverlayElement(overlay);  
      tList->AddElement(overlay);

      // set event handler, flip to top view at beginning
      {
         TGLViewer* v =  viewer();
         TEveLegoEventHandler* eh = new TEveLegoEventHandler("Lego",(TGWindow*)v->GetGLWidget(), (TObject*)v);
         v->SetEventHandler(eh);
         eh->Rotate(0,10000,kFALSE, kFALSE);
      }

      double ymax = lego->GetPhiMax();
      double ymin = lego->GetPhiMin();
      double xmax = lego->GetEtaMax();
      double xmin = lego->GetEtaMin();
      printf("lego range: xmin = %f xmax = %f, ymin = %f ymax = %f\n"
	     , xmin, xmax, ymin, ymax);

      // draw points for centroids
      TEveStraightLineSet *scposition = new TEveStraightLineSet("sc position");
      scposition->SetDepthTest(kFALSE);
      if (subdetId == EcalBarrel) {
         scposition->AddLine(i->caloPosition().eta(),
                             i->caloPosition().phi(),
                             0,
                             i->caloPosition().eta(),
                             i->caloPosition().phi(),
                             0);
      } else if (subdetId == EcalEndcap) {
         scposition->AddLine(i->caloPosition().x()*0.01,
                             i->caloPosition().y()*0.01,
                             0,
                             i->caloPosition().x()*0.01,
                             i->caloPosition().y()*0.01,
                             0);
      }
      scposition->AddMarker(0, 0.5);
      scposition->SetMarkerSize(2);
      scposition->SetMarkerColor(kBlue);
      tList->AddElement(scposition);
      TEveStraightLineSet *seedposition = new TEveStraightLineSet("seed position");
      seedposition->SetDepthTest(kFALSE);
      if (subdetId == EcalBarrel) {
	   seedposition->AddLine(i->superCluster()->seed()->position().eta(),
				 i->superCluster()->seed()->position().phi(),
				 0,
				 i->superCluster()->seed()->position().eta(),
				 i->superCluster()->seed()->position().phi(),
				 0);
      } else if (subdetId == EcalEndcap) {
	   seedposition->AddLine(i->superCluster()->seed()->position().x()*0.01,
				 i->superCluster()->seed()->position().y()*0.01,
				 0,
				 i->superCluster()->seed()->position().x()*0.01,
				 i->superCluster()->seed()->position().y()*0.01,
				 0);
      }
      seedposition->AddMarker(0, 0.5);
      seedposition->SetMarkerSize(2);
      seedposition->SetMarkerColor(kYellow);
      tList->AddElement(seedposition);

      // draw crosshairs for track intersections
      TEveStraightLineSet *trackpositionAtCalo =
         new TEveStraightLineSet("sc trackpositionAtCalo");
      trackpositionAtCalo->SetDepthTest(kFALSE);
      if (subdetId == EcalBarrel) {
	   trackpositionAtCalo->AddLine(i->TrackPositionAtCalo().eta(),
					ymin,
					0,
					i->TrackPositionAtCalo().eta(),
					ymax,
					0);
	   trackpositionAtCalo->AddLine(xmin,
					i->TrackPositionAtCalo().phi(),
					0,
					xmax,
					i->TrackPositionAtCalo().phi(),
					0);
      } else if (subdetId == EcalEndcap) {
	   trackpositionAtCalo->AddLine(i->TrackPositionAtCalo().x()*0.01,
					ymin,
					0,
					i->TrackPositionAtCalo().x()*0.01,
					ymax,
					0);
	   trackpositionAtCalo->AddLine(xmin,
					i->TrackPositionAtCalo().y()*0.01,
					0,
					xmax,
					i->TrackPositionAtCalo().y()*0.01,
					0);
      }
      trackpositionAtCalo->SetLineColor(kBlue);
      tList->AddElement(trackpositionAtCalo);
//       TEveStraightLineSet *pinposition = new TEveStraightLineSet("pin position");
//       if (subdetId == EcalBarrel) {
// 	   pinposition->AddLine((i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()),
// 				ymin, 
// 				0,
// 				(i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()),
// 				ymax,
// 				0);
// 	   pinposition->AddLine(xmin,
// 				(i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()),
// 				0,
// 				xmax,
// 				(i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()),
// 				0);
//       } else if (subdetId == EcalEndcap) {
// 	   pinposition->AddLine((i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
// 				ymin, 
// 				0,
// 				(i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
// 				ymax,
// 				0);
// 	   pinposition->AddLine(m_xmin,
// 				(i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
// 				0,
// 				m_xmax,
// 				(i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
// 				0);
//       }
//       printf("TrackPositionAtCalo: %f %f\n", 
// 	     i->TrackPositionAtCalo().eta(), i->TrackPositionAtCalo().phi());
//       printf("TrackPositionInner: %f %f\n",       
// 	     i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx(),
// 	     i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx());
//       printf("calo position %f, deltaEta %f, track position %f\n", 
// 	     i->caloPosition().eta(),
// 	     i->deltaEtaSuperClusterTrackAtVtx(),
// 	     i->TrackPositionAtCalo().eta());
//       pinposition->SetLineColor(kYellow);
//       tList->AddElement(pinposition);

      gEve->Redraw3D(kTRUE);
      return tList;
   }
   return tList;
}

void FWElectronDetailView::fillReducedData (const std::vector<DetId> &detids,
					    TEveCaloDataVec *data)
{
     if (detids.size() == 0)
	  return;
     DetId seed_crystal = *detids.begin();
     if (seed_crystal.subdetId() == EcalBarrel) {
	  
     } else if (seed_crystal.subdetId() == EcalEndcap) {

     }
}

void FWElectronDetailView::fillData (const std::vector<DetId> &detids,
                                     TEveCaloDataVec *data, 
                                     double phi_seed)
{
   for (std::vector<DetId>::const_iterator k = detids.begin();
        k != detids.end(); ++k) {
      double size = 50; // default size
      if (k->subdetId() == EcalBarrel) {
         if (m_barrel_hits != 0) {
            EcalRecHitCollection::const_iterator hit = 
               m_barrel_hits->find(*k);
            if (hit != m_barrel_hits->end()) {
               size = hit->energy();
            }
         }
      } else if (k->subdetId() == EcalEndcap) {
         if (m_endcap_hits != 0) {
            EcalRecHitCollection::const_iterator hit = 
               m_endcap_hits->find(*k);
            if (hit != m_endcap_hits->end()) {
               size = hit->energy();
            }
         }
      }
      const TGeoHMatrix *matrix = m_item->getGeom()->getMatrix(k->rawId());
      if ( matrix == 0 ) {
         printf("Warning: cannot get geometry for DetId: %d. Ignored.\n",k->rawId());
         continue;
      }
      const TVector3 v(matrix->GetTranslation()[0], 
                       matrix->GetTranslation()[1],
                       matrix->GetTranslation()[2]);
      // slice 1 is the non-seed clusters (which will show up in yellow)
      int slice = 1;
      if (find(seed_detids.begin(), seed_detids.end(), *k) != 
          seed_detids.end()) {
         // slice 0 is the seed cluster (which will show up in red)
         slice = 0;
      } 
      if (m_coordEtaPhi) {
         double phi = v.Phi();
         if (v.Phi() > phi_seed + M_PI)
            phi -= 2 * M_PI;
         if (v.Phi() < phi_seed - M_PI)
            phi += 2 * M_PI;
         data->AddTower(v.Eta() - 0.0172 / 2, v.Eta() + 0.0172 / 2, 
                        phi - 0.0172 / 2, phi + 0.0172 / 2);
         data->FillSlice(slice, size);
      } else if (k->subdetId() == EcalEndcap) {
         // temproary workaround for calo lego TwoPi periodic behaviour 
         // switch from cm to m
         data->AddTower((v.X() - 2.9 / 2)*0.01f, (v.X() + 2.9 / 2)*0.01f, 
                        (v.Y() - 2.9 / 2)*0.01f, (v.Y() + 2.9 / 2)*0.01f);
         data->FillSlice(slice, size);
      }
   }
   data->SetAxisFromBins(1e-2, 1e-2);

//      // add offset
//      Double_t etaMin, etaMax;
//      Double_t phiMin, phiMax;
//      data->GetEtaLimits(etaMin, etaMax);
//      data->GetPhiLimits(phiMin, phiMax);
//      Float_t offe = 0.1*(etaMax -etaMin);
//      Float_t offp = 0.1*(etaMax -etaMin);
//      data->AddTower(etaMin -offe, etaMax +offe, phiMin -offp , phiMax +offp);

   // set eta, phi axis title with symbol.ttf font
   if (detids.size() > 0 && detids.begin()->subdetId() == EcalEndcap) {
      data->GetEtaBins()->SetTitle("X[m]");
      data->GetPhiBins()->SetTitle("Y[m]");
   } else {
      data->GetEtaBins()->SetTitleFont(122);
      data->GetEtaBins()->SetTitle("h");
      data->GetPhiBins()->SetTitleFont(122);
      data->GetPhiBins()->SetTitle("f");
   }
}
     
TEveElementList *FWElectronDetailView::makeLabels (const reco::GsfElectron &electron)
{
   float_t x = 0.02;
   float   y = 0.83;
   float fontsize = latex()->GetTextSize()*0.5;
 
   TEveElementList *ret = new TEveElementList("electron labels");
   // summary
   if (electron.charge() > 0)
      latex()->DrawLatex(x, y, "charge = +1");
   else latex()->DrawLatex(x, y, "charge = -1");
   y -= fontsize;

   char summary[128];
   sprintf(summary, "%s = %.1f",
           "E_{T}", electron.caloEnergy() / cosh(electron.eta()));
   latex()->DrawLatex(x, y, summary);
   y -= fontsize;

   // E/p, H/E
   char hoe[128];
   sprintf(hoe, "E/p = %.2f",
           electron.eSuperClusterOverP());
   latex()->DrawLatex(x, y, hoe);
   y -= fontsize;
   sprintf(hoe, "H/E = %.3f", electron.hadronicOverEm());
   latex()->DrawLatex(x, y, hoe);
   y -= fontsize;
  
   // phi eta
   char ephi[30];
   sprintf(ephi, " #eta = %.2f #varphi=  %.2f", electron.eta(), electron.phi());
   latex()->DrawLatex(x, y, ephi);
   y -= fontsize;
 
   // delta phi/eta in
   char din[128];
   sprintf(din, "#Delta#eta_{in} = %.3f %16s = %.3f",
           electron.deltaEtaSuperClusterTrackAtVtx(),
           "#Delta#varphi_{in}", electron.deltaPhiSuperClusterTrackAtVtx());
   latex()->DrawLatex(x, y, din);
   y -= fontsize;

   // delta phi/eta out
   char dout[128];
   sprintf(dout, "#Delta#eta_{out} = %.3f %16s = %.3f",
           electron.deltaEtaSeedClusterTrackAtCalo(),
           "#Delta#varphi_{out}", electron.deltaPhiSeedClusterTrackAtCalo());
   latex()->DrawLatex(x, y, dout);
   y -= 2*fontsize;
   // legend

   latex()->DrawLatex(x, y, "#color[2]{+} track outer helix extrapolation");
   y -= fontsize;
   latex()->DrawLatex(x, y, "#color[4]{+} track inner helix extrapolation");
   y -= fontsize;
   latex()->DrawLatex(x, y, "#color[5]{#bullet} seed cluster centroid");
   y -= fontsize;
   latex()->DrawLatex(x, y, "#color[4]{#bullet} supercluster centroid");
   y -= fontsize;
   latex()->DrawLatex(x, y, "#color[2]{#Box} seed cluster");
   y -= fontsize;
   latex()->DrawLatex(x, y, "#color[5]{#Box} other clusters");
   // eta, phi axis or x, y axis?
   assert(electron.superCluster().isNonnull());
   bool is_endcap = false;
   if (electron.superCluster()->getHitsByDetId().size() > 0 &&
       electron.superCluster()->getHitsByDetId().begin()->subdetId() == EcalEndcap)
      is_endcap = true;

   return ret;
}

void FWElectronDetailView::getEcalCrystalsBarrel (std::vector<DetId> *vv, 
						  int ieta, int iphi)
{
     std::vector<DetId> &v = *vv;
     const int n_eta = 10;
     const int n_phi = 20;
     v.reserve((2 * n_eta + 1) * (2 * n_phi + 1));
     for (int i = ieta - n_eta; i < ieta + n_eta; ++i) {
	  for (int j = iphi - n_phi; j < iphi + n_phi; ++j) {
//             printf("pushing back (%d, %d)\n", i, j % 360);
	       if (EBDetId::validDetId(i, j % 360)) {
		    v.push_back(EBDetId(i, j % 360));
//                  printf("pushing back (%d, %d)\n", i, j % 360);
	       }
	  }
     }
}


void FWElectronDetailView::getEcalCrystalsEndcap (std::vector<DetId> *vv, 
						  int ix, int iy, int iz)
{
     std::vector<DetId> &v = *vv;
     const int n_x = 10;
     const int n_y = 10;
     v.reserve((2 * n_x + 1) * (2 * n_y + 1));
     for (int i = ix - n_x; i < ix + n_x; ++i) {
	  for (int j = iy - n_y; j < iy + n_y; ++j) {
	       if (EEDetId::validDetId(i, j, iz)) {
		    v.push_back(EEDetId(i, j, iz));
	       }
	  }
     }
}

REGISTER_FWDETAILVIEW(FWElectronDetailView);
