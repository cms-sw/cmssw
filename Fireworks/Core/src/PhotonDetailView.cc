// -*- C++ -*-
//
// Package:     Calo
// Class  :     PhotonDetailView
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: PhotonDetailView.cc,v 1.18 2008/11/26 15:34:15 amraktad Exp $
//

// system include files
#include "TClass.h"
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"
#include "TGeoBBox.h"
#include "TGeoArb8.h"
#include "TGeoTube.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEveBoxSet.h"
#include "TEveSceneInfo.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include <TEveLegoOverlay.h>
#include "TEveText.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveLegoEventHandler.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGLWidget.h"
#include "TGTextView.h"
#include "TRandom3.h"
#include "TH2.h"

// user include files
#include "Fireworks/Core/interface/PhotonDetailView.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
//
// constants, enums and typedefs
//
#define DRAW_LABELS_IN_SEPARATE_VIEW 1
#define USE_CALO_LEGO 1
//
// static data member definitions
//

//
// constructors and destructor
//
PhotonDetailView::PhotonDetailView()
{

}

// PhotonDetailView::PhotonDetailView(const PhotonDetailView& rhs)
// {
//    // do actual copying here;
// }

PhotonDetailView::~PhotonDetailView()
{
   resetCenter();
}

//
// member functions
//
void PhotonDetailView::build (TEveElementList **product, const FWModelId &id) 
{
     return build_projected(product, id);
}

void PhotonDetailView::build_projected (TEveElementList **product, 
					  const FWModelId &id) 
{
     m_item = id.item();
     // printf("calling PhotonDetailView::buildRhoZ\n");
     TEveElementList* tList = *product;
     if(0 == tList) {
	  tList =  new TEveElementList(m_item->name().c_str(),"Supercluster RhoZ",true);
	  *product = tList;
	  tList->SetMainColor(m_item->defaultDisplayProperties().color());
	  gEve->AddElement(tList);
     } else {
	  return;
// 	  tList->DestroyElements();
     }
     // get photons
//      resetCenter();
     using reco::PhotonCollection;
     const PhotonCollection *photons = 0;
     // printf("getting photons\n");
     m_item->get(photons);
     // printf("got photons\n");
     if (photons == 0) return;
     // printf("%d GSF photons\n", photons->size());
     // get rechits
/*
     const EcalRecHitCollection *hits = 0;
     const TClass *m_type  = TClass::GetClass("EcalRecHitCollection");
     ROOT::Reflex::Type dataType( ROOT::Reflex::Type::ByTypeInfo(*(m_type->GetTypeInfo())));
     assert(dataType != ROOT::Reflex::Type() );
     std::string wrapperName = std::string("edm::Wrapper<")+dataType.Name(ROOT::Reflex::SCOPED)+" >";
     //std::cout <<wrapperName<<std::endl;
     ROOT::Reflex::Type m_wrapperType = ROOT::Reflex::Type::ByName(wrapperName);

     void *tmp = 0;
     printf("getting rechits\n");
     const fwlite::Event *ev = m_item->getEvent();
     ev->getByLabel(m_wrapperType.TypeInfo(),
		    "ecalRecHit", "EcalRecHitsEB", 0, (void *)&tmp);
     printf("got rechits\n");
     hits = static_cast<const EcalRecHitCollection *>(tmp);
     if (hits == 0) {
	  std::cout <<"failed to get Ecal RecHits" << std::endl;
	  return;
     }
*/
     // printf("getting rechits\n");
     const fwlite::Event *ev = m_item->getEvent();
     fwlite::Handle<EcalRecHitCollection> h_barrel_hits;
     barrel_hits = 0;
     try {
	h_barrel_hits.getByLabel(*ev, "ecalRecHit", "EcalRecHitsEB");
	barrel_hits = h_barrel_hits.ptr();
     }
     catch (...) 
     {
	std::cout <<"no barrel ECAL rechits are available, "
	     "show only crystal location" << std::endl;
     }
     fwlite::Handle<EcalRecHitCollection> h_endcap_hits;
     endcap_hits = 0;
     try {
	h_endcap_hits.getByLabel(*ev, "ecalRecHit", "EcalRecHitsEE");
	endcap_hits = h_endcap_hits.ptr();
     }
     catch (...) 
     {
	std::cout <<"no endcap ECAL rechits are available, "
	     "show only crystal location" << std::endl;
     }
#if 0
     TH2F *ecalHist = new TH2F("ecalHist", "ecal hist", 170, -1.4, 1.4, 360, -M_PI, M_PI);
     TRandom3 r(42);
#endif
     if (const reco::Photon *i = &photons->at(id.index())) {
// 	  assert(i->gsfTrack().isNonnull());
	  assert(i->superCluster().isNonnull());
	  tList->AddElement(makeLabels(*i));
	  std::vector<DetId> detids = i->superCluster()->getHitsByDetId();
	  seed_detids = i->superCluster()->seed()->
	       getHitsByDetId();
	  //  const int subdetId = 
	  //   seed_detids.size() != 0 ? seed_detids.begin()->subdetId() : -1;
#if 0
	  for (int i = 0; i < 100; ++i) {
	       double eta = 0.01 * r.Rndm() - 0.4;
	       double phi = 0.1 * r.Rndm() + 0.3;
	       double E = 100 * r.Rndm();
	       ecalHist->Fill(eta, phi, E);
	  }
	  TEveCaloDataHist* hd = new TEveCaloDataHist();
	  Int_t s = hd->AddHistogram(ecalHist);
	  hd->RefSliceInfo(s).Setup("ECAL", 0.3, kRed);
	  TEveCalo3D* calo3d = new TEveCalo3D(hd);
	  calo3d->SetBarrelRadius(129);
	  calo3d->SetEndCapPos(300);
	  tList->AddElement(calo3d);
	  tList->AddElement(makeLabels(*i));
#endif
	  // vector for the ECAL crystals
	  TEveCaloDataVec* data = new TEveCaloDataVec(2);
	  // one slice for the seed cluster (red) and one for the
	  // other clusters
	  data->RefSliceInfo(0).Setup("seed cluster", 0.3, kRed);
	  data->RefSliceInfo(1).Setup("other clusters", 0.1, kYellow);
	  // now fill
#if 1
	  fillData(detids, data);
#else
	  data->AddTower(0.12, 0.14, 0.45, 0.47);
	  data->FillSlice(0, 12);
	  data->FillSlice(1, 3);
     
	  data->AddTower(0.125, 0.145, 0.43, 0.45);
	  data->FillSlice(0, 4);
	  data->FillSlice(1, 7);
	  
	  data->AddTower(0.10, 0.12, 0.45, 0.47);
	  data->FillSlice(0, 6);
	  data->FillSlice(1, 0);
	  
	  data->SetEtaBins(new TAxis(10, 0.08, 0.16));
	  data->SetPhiBins(new TAxis(10, 0.40, 0.50));
#endif  
	  data->DataChanged();
	  
	  // lego
	  
	  TEveCaloLego* lego = new TEveCaloLego(data);
          lego->SetAutoRebin(kFALSE);
	  lego->SetPlaneColor(kBlue-5);

	  // tempoary solution until we get pointer to gl viewer
	  lego->SetProjection(TEveCaloLego::k3D);
	  //  TEveLegoEventHandler *leh = new TEveLegoEventHandler("Lego EH", v->GetGLWidget(), lego->GetObject(), "fooo");
	  //  ve->SetEventHandler(leh);

	  lego->SetName("PhotonDetail Lego");
	  printf("%f %f, %f %f\n", x_min, x_max, y_min, y_max);
	  tList->AddElement(lego);
	  
	  // overlay lego
	  
	  TEveLegoOverlay* overlay = new TEveLegoOverlay();
	  overlay->SetCaloLego(lego);
	  TGLViewer* v = gEve->GetGLViewer(); // Default
	  v->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
	  v->AddOverlayElement(overlay);
	  tList->AddElement(overlay);
	  gEve->Redraw3D(kTRUE);
	  
	  return;
     }
}

void PhotonDetailView::fillData (const std::vector<DetId> &detids,
				   TEveCaloDataVec *data)
{
     x_min = 999;
     x_max = -999;
     y_min = 999;
     y_max = -999;
     for (std::vector<DetId>::const_iterator k = detids.begin();
	  k != detids.end(); ++k) {
	  double size = 50; // default size
	  if (k->subdetId() == EcalBarrel) {
	       if (barrel_hits != 0) {
		    EcalRecHitCollection::const_iterator hit = 
			 barrel_hits->find(*k);
		    if (hit != barrel_hits->end()) {
			 size = hit->energy();
		    }
	       }
	  } else if (k->subdetId() == EcalEndcap) {
	       if (endcap_hits != 0) {
		    EcalRecHitCollection::const_iterator hit = 
			 endcap_hits->find(*k);
		    if (hit != endcap_hits->end()) {
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
	  int slice = 1;
	  if (find(seed_detids.begin(), seed_detids.end(), *k) != 
	      seed_detids.end()) {
	       slice = 0;
	  } 
	  if (k->subdetId() == EcalBarrel) {
	       if (v.Eta() < x_min)
		    x_min = v.Eta();
	       if (v.Eta() > x_max)
		    x_max = v.Eta();
	       if (v.Phi() < y_min)
		    y_min = v.Phi();
	       if (v.Phi() > y_max)
		    y_max = v.Phi();
	       data->AddTower(v.Eta() - 0.0174 / 2, v.Eta() + 0.0174 / 2, 
			      v.Phi() - 0.0174 / 2, v.Phi() + 0.0174 / 2);
	       data->FillSlice(slice, size);
	  } else if (k->subdetId() == EcalEndcap) {
	       if (v.X() < x_min)
		    x_min = v.X();
	       if (v.X() > x_max)
		    x_max = v.X();
	       if (v.Y() < y_min)
		    y_min = v.Y();
	       if (v.Y() > y_max)
		    y_max = v.Y();
	       data->AddTower(v.X() - 2.2 / 2, v.X() + 2.2 / 2, 
			      v.Y() - 2.2 / 2, v.Y() + 2.2 / 2);
	       data->FillSlice(slice, size);
	  }
     }
     data->SetAxisFromBins(1e-2, 1e-2);
}
     
TEveElementList *PhotonDetailView::makeLabels (
     const reco::Photon &photon) 
{
     TEveElementList *ret = new TEveElementList("photon labels");
#if DRAW_LABELS_IN_SEPARATE_VIEW
     // title
     text_view->AddLine("Photon detailed view"); 
     text_view->AddLine("");
     // summary
     char summary[128];
     sprintf(summary, "%s = %.1f GeV %10s = %.2f %10s = %.2f",
	     "ET", photon.et(), 
	     "eta", photon.eta(), 
	     "phi", photon.phi());
     text_view->AddLine(summary);
     // E/p, H/E
     char hoe[128];
     sprintf(hoe, "H/E = %.3f",
	     photon.hadronicOverEm());
     text_view->AddLine(hoe);
     // legend
     text_view->AddLine("");
     text_view->AddLine("      red point: seed cluster centroid");
     text_view->AddLine("     blue point: supercluster centroid");
     text_view->AddLine("   red crystals: seed cluster");
     text_view->AddLine("yellow crystals: other clusters");
#else
     // title
     TEveText* t = new TEveText("Photon detailed view");
     t->PtrMainTrans()->MoveLF(1, rotation_center[0] + 5);
     t->PtrMainTrans()->MoveLF(2, rotation_center[1] + 10);
     t->SetMainColor((Color_t)(kWhite));
     t->SetFontSize(16);
     t->SetFontFile(8);
     t->SetLighting(kTRUE);
     ret->AddElement(t);
     // summary
     char summary[128];
     sprintf(summary, "ET = %.1f GeV        eta = %.2f        phi = %.2f",
	     photon.caloEnergy() / cosh(photon.eta()), 
	     photon.eta(), photon.phi());
     t = new TEveText(summary);
     t->PtrMainTrans()->MoveLF(1, rotation_center[0] + 4.5);
     t->PtrMainTrans()->MoveLF(2, rotation_center[1] + 10);
     t->SetMainColor((Color_t)(kWhite));
     t->SetFontSize(12);
     t->SetFontFile(8);
     t->SetLighting(kTRUE);
     ret->AddElement(t);
     // E/p, H/E
     char hoe[128];
     sprintf(hoe, "E/p = %.2f        H/E = %.3f",
	     photon.eSuperClusterOverP(), photon.hadronicOverEm());
     t = new TEveText(hoe);
     t->PtrMainTrans()->MoveLF(1, rotation_center[0] + 4.0);
     t->PtrMainTrans()->MoveLF(2, rotation_center[1] + 10);
     t->SetMainColor((Color_t)(kWhite));
     t->SetFontSize(12);
     t->SetFontFile(8);
     t->SetLighting(kTRUE);
     ret->AddElement(t);
     // delta phi/eta in 
     char din[128];
     sprintf(din, "delta eta in = %.3f        delta phi in = %.3f",
	     photon.deltaEtaSuperClusterTrackAtVtx(),
	     photon.deltaPhiSuperClusterTrackAtVtx());
     t = new TEveText(din);
     t->PtrMainTrans()->MoveLF(1, rotation_center[0] + 3.5);
     t->PtrMainTrans()->MoveLF(2, rotation_center[1] + 10);
     t->SetMainColor((Color_t)(kWhite));
     t->SetFontSize(12);
     t->SetFontFile(8);
     t->SetLighting(kTRUE);
     ret->AddElement(t);
     // delta phi/eta out 
     char dout[128];
     sprintf(dout, "delta eta out = %.3f        delta phi out = %.3f",
	     photon.deltaEtaSeedClusterTrackAtCalo(),
	     photon.deltaPhiSeedClusterTrackAtCalo());
     t = new TEveText(dout);
     t->PtrMainTrans()->MoveLF(1, rotation_center[0] + 3.0);
     t->PtrMainTrans()->MoveLF(2, rotation_center[1] + 10);
     t->SetMainColor((Color_t)(kWhite));
     t->SetFontSize(12);
     t->SetFontFile(8);
     t->SetLighting(kTRUE);
     ret->AddElement(t);
#endif
#if 0
     // eta, phi axis
     TEveLine *eta_line = new TEveLine;
     eta_line->SetNextPoint(rotation_center[0] - 15, rotation_center[1] - 40, 0);
     eta_line->SetNextPoint(rotation_center[0] - 10, rotation_center[1] - 40, 0);
     eta_line->SetLineColor((Color_t)kWhite);
     ret->AddElement(eta_line);
     TEveText *tt = new TEveText("eta");
     tt->PtrMainTrans()->MoveLF(1, rotation_center[0] - 9);
     tt->PtrMainTrans()->MoveLF(2, rotation_center[1] - 40);
     ret->AddElement(tt);
     TEveLine *phi_line = new TEveLine;
     phi_line->SetNextPoint(rotation_center[0] - 15, rotation_center[1] - 40, 0);
     phi_line->SetNextPoint(rotation_center[0] - 15, rotation_center[1] - 35, 0);
     phi_line->SetLineColor((Color_t)kWhite);
     ret->AddElement(phi_line);
     tt = new TEveText("phi");
     tt->PtrMainTrans()->MoveLF(1, rotation_center[0] - 15);
     tt->PtrMainTrans()->MoveLF(2, rotation_center[1] - 34);
     ret->AddElement(tt);
#endif
     return ret;
}
