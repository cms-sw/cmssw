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
// $Id: FWElectronDetailView.cc,v 1.11 2009/03/19 14:06:21 jmuelmen Exp $
//

// system include files
#include "TGTextView.h"
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
   m_barrel_hits(0),
   m_endcap_hits(0)
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
         "show only crystal location" << std::endl;
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
         "show only crystal location" << std::endl;
   }
   if (const reco::GsfElectron *i = iElectron) {
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
         rotationCenter()[0] = i->superCluster()->position().x();
         rotationCenter()[1] = i->superCluster()->position().y();
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

      // debug (see also overlay)
      m_ymax = lego->GetPhiMax();
      m_ymin = lego->GetPhiMin();
      m_xmax = lego->GetEtaMax();
      m_xmin = lego->GetEtaMin();
      printf("lego range: xmin = %f xmax = %f, ymin = %f ymax = %f\n"
	     , m_xmin, m_xmax, m_ymin, m_ymax);

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
         scposition->AddLine(i->caloPosition().x(),
                             i->caloPosition().y(),
                             0,
                             i->caloPosition().x(),
                             i->caloPosition().y(),
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
	   seedposition->AddLine(i->superCluster()->seed()->position().x(),
				 i->superCluster()->seed()->position().y(),
				 0,
				 i->superCluster()->seed()->position().x(),
				 i->superCluster()->seed()->position().y(),
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
					m_ymin,
					0,
					i->TrackPositionAtCalo().eta(),
					m_ymax,
					0);
	   trackpositionAtCalo->AddLine(m_xmin,
					i->TrackPositionAtCalo().phi(),
					0,
					m_xmax,
					i->TrackPositionAtCalo().phi(),
					0);
      } else if (subdetId == EcalEndcap) {
	   trackpositionAtCalo->AddLine(i->TrackPositionAtCalo().x(),
					m_ymin,
					0,
					i->TrackPositionAtCalo().x(),
					m_ymax,
					0);
	   trackpositionAtCalo->AddLine(m_xmin,
					i->TrackPositionAtCalo().y(),
					0,
					m_xmax,
					i->TrackPositionAtCalo().y(),
					0);
      }
      trackpositionAtCalo->SetLineColor(kBlue);
      tList->AddElement(trackpositionAtCalo);
      TEveStraightLineSet *pinposition = new TEveStraightLineSet("pin position");
      if (subdetId == EcalBarrel) {
	   pinposition->AddLine((i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()),
				m_ymin, 
				0,
				(i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()),
				m_ymax,
				0);
	   pinposition->AddLine(m_xmin,
				(i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()),
				0,
				m_xmax,
				(i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()),
				0);
      } else if (subdetId == EcalEndcap) {
// 	   pinposition->AddLine((i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
// 				m_ymin, 
// 				0,
// 				(i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
// 				m_ymax,
// 				0);
// 	   pinposition->AddLine(m_xmin,
// 				(i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
// 				0,
// 				m_xmax,
// 				(i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
// 				0);
      }
      pinposition->SetLineColor(kYellow);
      tList->AddElement(pinposition);

      gEve->Redraw3D(kTRUE);
      return tList;
   }
   return tList;
}

void FWElectronDetailView::fillData (const std::vector<DetId> &detids,
				   TEveCaloDataVec *data, 
				   double phi_seed)
{
     m_xmin = 999;
     m_xmax = -999;
     m_ymin = 999;
     m_ymax = -999;
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
	  if (k->subdetId() == EcalBarrel) {
	       if (v.Eta() < m_xmin)
		    m_xmin = v.Eta();
	       if (v.Eta() > m_xmax)
		    m_xmax = v.Eta();
	       if (v.Phi() < m_ymin)
		    m_ymin = v.Phi();
	       if (v.Phi() > m_ymax)
		    m_ymax = v.Phi();
	       double phi = v.Phi();
	       if (v.Phi() > phi_seed + M_PI)
		    phi -= 2 * M_PI;
	       if (v.Phi() < phi_seed - M_PI)
		    phi += 2 * M_PI;
	       data->AddTower(v.Eta() - 0.0172 / 2, v.Eta() + 0.0172 / 2, 
			      phi - 0.0172 / 2, phi + 0.0172 / 2);
	       data->FillSlice(slice, size);
	  } else if (k->subdetId() == EcalEndcap) {
	       if (v.X() < m_xmin)
		    m_xmin = v.X();
	       if (v.X() > m_xmax)
		    m_xmax = v.X();
	       if (v.Y() < m_ymin)
		    m_ymin = v.Y();
	       if (v.Y() > m_ymax)
		    m_ymax = v.Y();
	       data->AddTower(v.X() - 2.2 / 2, v.X() + 2.2 / 2, 
			      v.Y() - 2.2 / 2, v.Y() + 2.2 / 2);
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
	  data->GetEtaBins()->SetTitle("X[cm]");
	  data->GetPhiBins()->SetTitle("Y[cm]");
     } else {
	  data->GetEtaBins()->SetTitleFont(122);
	  data->GetEtaBins()->SetTitle("h");
	  data->GetPhiBins()->SetTitleFont(122);
	  data->GetPhiBins()->SetTitle("f");
     }
}
     
TEveElementList *FWElectronDetailView::makeLabels (const reco::GsfElectron &electron)
{
   TEveElementList *ret = new TEveElementList("electron labels");
   // title
   textView()->AddLine("Electron detailed view");
   textView()->AddLine("");
   // summary
   if (electron.charge() > 0)
      textView()->AddLine("charge = +1");
   else textView()->AddLine("charge = -1");
   char summary[128];
   sprintf(summary, "%s = %.1f GeV %10s = %.2f %10s = %.2f",
           "ET", electron.caloEnergy() / cosh(electron.eta()),
           "eta", electron.eta(),
           "phi", electron.phi());
   textView()->AddLine(summary);
   // E/p, H/E
   char hoe[128];
   sprintf(hoe, "E/p = %.2f %13s = %.3f",
           electron.eSuperClusterOverP(),
           "H/E", electron.hadronicOverEm());
   textView()->AddLine(hoe);
   // delta phi/eta in
   char din[128];
   sprintf(din, "delta eta in = %.3f %16s = %.3f",
           electron.deltaEtaSuperClusterTrackAtVtx(),
           "delta phi in", electron.deltaPhiSuperClusterTrackAtVtx());
   textView()->AddLine(din);
   // delta phi/eta out
   char dout[128];
   sprintf(dout, "delta eta out = %.3f %16s = %.3f",
           electron.deltaEtaSeedClusterTrackAtCalo(),
           "delta phi out", electron.deltaPhiSeedClusterTrackAtCalo());
   textView()->AddLine(dout);
   // legend
   textView()->AddLine("");
   textView()->AddLine("      red cross: track outer helix extrapolation");
   textView()->AddLine("     blue cross: track inner helix extrapolation");
   textView()->AddLine("      red point: seed cluster centroid");
   textView()->AddLine("     blue point: supercluster centroid");
   textView()->AddLine("   red crystals: seed cluster");
   textView()->AddLine("yellow crystals: other clusters");
   // eta, phi axis or x, y axis?
   assert(electron.superCluster().isNonnull());
   bool is_endcap = false;
   if (electron.superCluster()->getHitsByDetId().size() > 0 &&
       electron.superCluster()->getHitsByDetId().begin()->subdetId() == EcalEndcap)
      is_endcap = true;
   return ret;
}

TEveElementList *FWElectronDetailView::getEcalCrystalsBarrel (
   const DetIdToMatrix &geo,
   double eta, double phi,
   int n_eta, int n_phi)
{
   std::vector<DetId> v;
   int ieta = (int)rint(eta / 1.74e-2);
   // black magic for phi
   int iphi = (int)rint(phi / 1.74e-2);
   if (iphi < 0)
      iphi = 360 + iphi;
   iphi += 10;
   for (int i = ieta - n_eta; i < ieta + n_eta; ++i) {
      for (int j = iphi - n_phi; j < iphi + n_phi; ++j) {
//             printf("pushing back (%d, %d)\n", i, j % 360);
         if (EBDetId::validDetId(i, j % 360)) {
            v.push_back(EBDetId(i, j % 360));
//                  printf("pushing back (%d, %d)\n", i, j % 360);
         }
      }
   }
   return getEcalCrystalsBarrel(geo, v);
}

TEveElementList *FWElectronDetailView::getEcalCrystalsBarrel (
   const DetIdToMatrix &geo,
   const std::vector<DetId> &detids)
{
   TEveElementList *ret = new TEveElementList("Ecal barrel crystals");
   for (std::vector<DetId>::const_iterator k = detids.begin();
        k != detids.end(); ++k) {
      const TGeoHMatrix *matrix = m_item->getGeom()->
                                  getMatrix(k->rawId());
      const TVector3 v(matrix->GetTranslation()[0],
                       matrix->GetTranslation()[1],
                       matrix->GetTranslation()[2]);
//        printf("trying to add DetId %d... ", k->rawId());
      if (k->subdetId() != EcalBarrel) {
//             printf("not in barrel\n");
         continue;
      }
//        printf("adding\n");
      const double scale = 100;
      float rgba[4] = { 1, 1, 0, 1 };
      TGeoBBox *box = new TGeoBBox(0.48 * 0.0172 * scale,
                                   0.48 * 0.0172 * scale,
                                   0.01, 0);
      TEveTrans t_box;
      t_box.SetPos(v.Eta() * scale,
                   v.Phi() * scale,
                   -0.11);
      TEveGeoShape *egs = new TEveGeoShape("EB crystal");
      egs->SetShape(box);
      egs->SetTransMatrix(t_box.Array());
      egs->SetMainColorRGB(rgba[1], rgba[2], rgba[3]);
      egs->SetMainTransparency(80);
      ret->AddElement(egs);
   }
   return ret;
}

TEveElementList *FWElectronDetailView::getEcalCrystalsEndcap (
   const DetIdToMatrix &geo,
   double x, double y, int iz,
   int n_x, int n_y)
{
   std::vector<DetId> v(n_x * n_y);
   int ix = (int)rint(x / 2.9) + 50;
   int iy = (int)rint(y / 2.9) + 50;
   for (int i = ix - n_x; i < ix + n_x; ++i) {
      for (int j = iy - n_y; j < iy + n_y; ++j) {
//             printf("pushing back (%d, %d)\n", i, j % 360);
         if (EEDetId::validDetId(i, j, iz)) {
            v.push_back(EEDetId(i, j, iz));
//                  printf("pushing back (%d, %d)\n", i, j % 360);
         }
      }
   }
   return getEcalCrystalsEndcap(geo, v);
}

TEveElementList *FWElectronDetailView::getEcalCrystalsEndcap (
   const DetIdToMatrix &geo,
   const std::vector<DetId> &detids)
{
   TEveElementList *ret = new TEveElementList("Ecal endcap crystals");
   for (std::vector<DetId>::const_iterator k = detids.begin();
        k != detids.end(); ++k) {
      const TGeoHMatrix *matrix = m_item->getGeom()->
                                  getMatrix(k->rawId());
      if (matrix == 0)
         continue;
      const TVector3 v(matrix->GetTranslation()[0],
                       matrix->GetTranslation()[1],
                       matrix->GetTranslation()[2]);
//        printf("trying to add DetId %d... ", k->rawId());
      if (k->subdetId() != EcalEndcap) {
//             printf("not in barrel\n");
         continue;
      }
//        printf("adding\n");
      const double scale = 1;
      float rgba[4] = { 1, 1, 0, 1 };
      TGeoBBox *box = new TGeoBBox(0.48 * 2.9 * scale,
                                   0.48 * 2.9 * scale,
                                   0.01, 0);
      TEveTrans t_box;
      t_box.SetPos(v.X() * scale,
                   v.Y() * scale,
                   -0.11);
      TEveGeoShape *egs = new TEveGeoShape("EEcrystal");
      egs->SetShape(box);
      egs->SetTransMatrix(t_box.Array());
      egs->SetMainColorRGB(rgba[0], rgba[1], rgba[2]);
      egs->SetMainTransparency(80);
      ret->AddElement(egs);
   }
   return ret;
}

REGISTER_FWDETAILVIEW(FWElectronDetailView);

