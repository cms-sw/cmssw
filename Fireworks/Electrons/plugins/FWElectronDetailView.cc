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
// $Id: FWElectronDetailView.cc,v 1.3 2009/03/17 17:25:51 jmuelmen Exp $
//

// system include files
#include "Rtypes.h"
#include "TClass.h"
#include "TEveGeoNode.h"
#include "TGeoBBox.h"
#include "TGeoArb8.h"
#include "TGeoTube.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEveBoxSet.h"
#include "TEveSceneInfo.h"
#define private public
#define protected public
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TEveLegoEventHandler.h"
#include <TEveCaloLegoOverlay.h>
#undef private
#undef protected
#include "TEveText.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGTextView.h"
#include "TEveBoxSet.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"

#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"


class FWElectronDetailView : public FWDetailView<reco::GsfElectron> {

public:
   FWElectronDetailView();
   virtual ~FWElectronDetailView();

   virtual TEveElement* build (const FWModelId &id, const reco::GsfElectron*);

protected:
   void setItem (const FWEventItem *iItem) {
      m_item = iItem;
   }
   TEveElement* build_projected (const FWModelId &id, const reco::GsfElectron*);
   void getCenter( Double_t* vars )
   {
      vars[0] = rotationCenter()[0];
      vars[1] = rotationCenter()[1];
      vars[2] = rotationCenter()[2];
   }
   TEveElementList *makeLabels (const reco::GsfElectron &);
   TEveElementList *getEcalCrystalsBarrel (const class DetIdToMatrix &,
                                           const std::vector<class DetId> &);
   TEveElementList *getEcalCrystalsBarrel (const class DetIdToMatrix &,
                                           double eta, double phi,
                                           int n_eta = 5, int n_phi = 10);
   TEveElementList *getEcalCrystalsEndcap (const class DetIdToMatrix &,
                                           const std::vector<class DetId> &);
   TEveElementList *getEcalCrystalsEndcap (const class DetIdToMatrix &,
                                           double x, double y, int iz,
                                           int n_x = 5, int n_y = 5);

     void fillData (const std::vector<DetId> &detids,
					TEveCaloDataVec *data, 
					double phi_seed);
     void rescale (TEveTrans *trans, double x_min, double x_max, 
				       double y_min, double y_max);

private:
   FWElectronDetailView(const FWElectronDetailView&); // stop default
   const FWElectronDetailView& operator=(const FWElectronDetailView&); // stop default

   // ---------- member data --------------------------------
   const FWEventItem* m_item;
   void resetCenter() {
      rotationCenter()[0] = 0;
      rotationCenter()[1] = 0;
      rotationCenter()[2] = 0;
   }
     const EcalRecHitCollection *barrel_hits;
     const EcalRecHitCollection *endcap_hits;
     std::vector<DetId> seed_detids;

     double x_min;
     double x_max;
     double y_min;
     double y_max;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWElectronDetailView::FWElectronDetailView()
{

}

// FWElectronDetailView::FWElectronDetailView(const FWElectronDetailView& rhs)
// {
//    // do actual copying here;
// }

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
   if(0==iElectron) { return 0;}
   m_item = id.item();
   // printf("calling FWElectronDetailView::buildRhoZ\n");
   TEveElementList* tList =  new TEveElementList(m_item->name().c_str(),"Supercluster RhoZ",true);
   tList->SetMainColor(m_item->defaultDisplayProperties().color());
   gEve->AddElement(tList);
   // get electrons
   resetCenter();
   // get rechits
   const fwlite::Event *ev = m_item->getEvent();
   fwlite::Handle<EcalRecHitCollection> h_barrel_hits;
   const EcalRecHitCollection* barrel_hits(0);
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
   float rgba[4] = { 1, 0, 0, 1 };
   if (const reco::GsfElectron *i = iElectron) {
      assert(i->gsfTrack().isNonnull());
      assert(i->superCluster().isNonnull());
      tList->AddElement(makeLabels(*i));
      std::vector<DetId> detids = i->superCluster()->getHitsByDetId();
      seed_detids = i->superCluster()->seed()->
	   getHitsByDetId();
      const unsigned int subdetId = 
	   seed_detids.size() != 0 ? seed_detids.begin()->subdetId() : 0;
      //  const int subdetId = 
      //   seed_detids.size() != 0 ? seed_detids.begin()->subdetId() : -1;
      const double scale = 1;
      if (subdetId == EcalBarrel) {
         rotationCenter()[0] = i->superCluster()->position().eta() * scale;
         rotationCenter()[1] = i->superCluster()->position().phi() * scale;
         rotationCenter()[2] = 0;
      } else if (subdetId == EcalEndcap) {
         rotationCenter()[0] = i->superCluster()->position().x() * scale;
         rotationCenter()[1] = i->superCluster()->position().y() * scale;
         rotationCenter()[2] = 0;
      }
//        rotationCenter()[0] = i->TrackPositionAtCalo().x();
//        rotationCenter()[1] = i->TrackPositionAtCalo().y();
//        rotationCenter()[2] = i->TrackPositionAtCalo().z();
      TEvePointSet *scposition =
         new TEvePointSet("sc position", 1);
      if (subdetId == EcalBarrel) {
         scposition->SetNextPoint(i->caloPosition().eta() * scale,
                                  i->caloPosition().phi() * scale,
                                  1);
      } else if (subdetId == EcalEndcap) {
         scposition->SetNextPoint(i->caloPosition().x() * scale,
                                  i->caloPosition().y() * scale,
                                  1);
      }
      scposition->SetMarkerStyle(28);
      scposition->SetMarkerSize(0.25);
      scposition->SetMarkerColor(kBlue);
      tList->AddElement(scposition);
      TEvePointSet *seedposition =
         new TEvePointSet("seed position", 1);
      if (subdetId == EcalBarrel) {
         seedposition->SetNextPoint(i->superCluster()->seed()->position().eta() * scale,
                                    i->superCluster()->seed()->position().phi() * scale,
                                    1);
      } else if (subdetId == EcalEndcap) {
         seedposition->SetNextPoint(i->superCluster()->seed()->position().x() * scale,
                                    i->superCluster()->seed()->position().y() * scale,
                                    1);
	 printf("seed cluster position: %f %f\n",
		i->superCluster()->seed()->position().eta(),
		i->superCluster()->seed()->position().phi());
      }
      seedposition->SetMarkerStyle(28);
      seedposition->SetMarkerSize(0.25);
      seedposition->SetMarkerColor(kRed);
      tList->AddElement(seedposition);
      TEveLine *trackpositionAtCalo =
         new TEveLine("sc trackpositionAtCalo");
      if (subdetId == EcalBarrel) {
         trackpositionAtCalo->SetNextPoint(i->TrackPositionAtCalo().eta() * scale,
                                           rotationCenter()[1] - 0.5,
                                           0);
         trackpositionAtCalo->SetNextPoint(i->TrackPositionAtCalo().eta() * scale,
                                           rotationCenter()[1] + 0.5,
                                           0);
      } else if (subdetId == EcalEndcap) {
         trackpositionAtCalo->SetNextPoint(i->TrackPositionAtCalo().x() * scale,
                                           rotationCenter()[1] - 0.5,
                                           0);
         trackpositionAtCalo->SetNextPoint(i->TrackPositionAtCalo().x() * scale,
                                           rotationCenter()[1] + 0.5,
                                           0);
      }
      trackpositionAtCalo->SetLineColor(kBlue);
      tList->AddElement(trackpositionAtCalo);
      trackpositionAtCalo = new TEveLine("sc trackpositionAtCalo");
      if (subdetId == EcalBarrel) {
         trackpositionAtCalo->SetNextPoint(rotationCenter()[0] - 0.5,
                                           i->TrackPositionAtCalo().phi() * scale,
                                           0);
         trackpositionAtCalo->SetNextPoint(rotationCenter()[0] + 0.5,
                                           i->TrackPositionAtCalo().phi() * scale,
                                           0);
      } else if (subdetId == EcalEndcap) {
         trackpositionAtCalo->SetNextPoint(rotationCenter()[0] - 0.5,
                                           i->TrackPositionAtCalo().y() * scale,
                                           0);
         trackpositionAtCalo->SetNextPoint(rotationCenter()[0] + 0.5,
                                           i->TrackPositionAtCalo().y() * scale,
                                           0);
      }
      trackpositionAtCalo->SetLineColor(kBlue);
      tList->AddElement(trackpositionAtCalo);
      TEveLine *pinposition =
         new TEveLine("pin position", 1);
      if (subdetId == EcalBarrel) {
         pinposition->SetNextPoint((i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
                                   rotationCenter()[1] - 0.5,
                                   0);
         pinposition->SetNextPoint((i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
                                   rotationCenter()[1] + 0.5,
                                   0);
      } else if (subdetId == EcalEndcap) {
         pinposition->SetNextPoint((i->caloPosition().x() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
                                   rotationCenter()[1] - 0.5,
                                   0);
         pinposition->SetNextPoint((i->caloPosition().x() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
                                   rotationCenter()[1] + 0.5,
                                   0);
      }
      pinposition->SetMarkerStyle(28);
      pinposition->SetLineColor(kRed);
      tList->AddElement(pinposition);
      pinposition = new TEveLine("pin position", 1);
      if (subdetId == EcalBarrel) {
         pinposition->SetNextPoint(rotationCenter()[0] - 0.5,
                                   (i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
                                   0);
         pinposition->SetNextPoint(rotationCenter()[0] + 0.5,
                                   (i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
                                   0);
      } else if (subdetId == EcalEndcap) {
         pinposition->SetNextPoint(rotationCenter()[0] - 0.5,
                                   (i->caloPosition().y() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
                                   0);
         pinposition->SetNextPoint(rotationCenter()[0] + 0.5,
                                   (i->caloPosition().y() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
                                   0);
      }
      pinposition->SetMarkerStyle(28);
      pinposition->SetLineColor(kRed);
      tList->AddElement(pinposition);
      // vector for the ECAL crystals
      TEveCaloDataVec* data = new TEveCaloDataVec(2);
      // one slice for the seed cluster (red) and one for the
      // other clusters
      data->RefSliceInfo(0).Setup("seed cluster", 0.3, kRed);
      data->RefSliceInfo(1).Setup("other clusters", 0.1, kYellow);
      // now fill
      fillData(detids, data, i->superCluster()->seed()->position().phi());
      data->DataChanged();
      data->fMaxValEt *= 4;
      data->fMaxValE *= 4;
         
      // lego
         
      TEveCaloLego* lego = new TEveCaloLego(data);
      lego->SetAutoRebin(kFALSE);
      lego->SetPlaneColor(kBlue-5);

      // tempoary solution until we get pointer to gl viewer
//       lego->SetProjection(TEveCaloLego::k3D);
      lego->Set2DMode(TEveCaloLego::kValSize);
      lego->SetName("ElectronDetail Lego");
//       lego->SetMainTransparency(5);
      gEve->AddElement(lego, tList);

      // scale and translate  
      lego->InitMainTrans();
//       lego->RefMainTrans().SetPos(0.5 * (x_min + x_max),
//                                   0.5 * (y_min + y_max),
//                                   0);
//       lego->RefMainTrans().SetScale(x_max - x_min, 
//                                     x_max - x_min,
//                                     1);

      // overlay lego
         
      TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
      overlay->SetShowPlane(kFALSE);
      overlay->SetShowPerspective(kFALSE);
      overlay->SetCaloLego(lego);
      TGLViewer* v = viewer();
      v->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
      v->AddOverlayElement(overlay);
      v->SetEventHandler(new TEveLegoEventHandler("Lego",(TGWindow*)v->GetGLWidget(), (TObject*)v));
      tList->AddElement(overlay);
      gEve->Redraw3D(kTRUE);

      double y_max = lego->GetPhiMax();
      double y_min = lego->GetPhiMin();
      double x_max = lego->GetEtaMax();
      double x_min = lego->GetEtaMin();
      printf("crystal range: xmin = %f xmax = %f, ymin = %f ymax = %f\n", 
	     x_min, x_max, y_min, y_max);
      printf("lego range: xmin = %f xmax = %f, ymin = %f ymax = %f\n"
	     , x_min, x_max, y_min, y_max);

      // scale all our lines and points to match the lego
      rescale(&scposition->RefMainTrans(), x_min, x_max, y_min, y_max);
      rescale(&seedposition->RefMainTrans(), x_min, x_max, y_min, y_max);
      rescale(&pinposition->RefMainTrans(), x_min, x_max, y_min, y_max);
//          rescale(&pinposition2->RefMainTrans(), x_min, x_max, y_min, y_max);
      rescale(&trackpositionAtCalo->RefMainTrans(), x_min, x_max, y_min, y_max);
//          rescale(&trackpositionAtCalo2->RefMainTrans(), x_min, x_max, y_min, y_max);
         
//          tList->AddElement(pinposition);
//          tList->AddElement(pinposition2);

      gEve->Redraw3D(kTRUE);
      return tList;
   }
   return tList;
}

void FWElectronDetailView::rescale (TEveTrans *trans, double x_min, double x_max, 
				  double y_min, double y_max)
{
     trans->SetScale(1 / (x_max - x_min), 
		     1 / (x_max - x_min),
		     1);
     trans->SetPos(-0.5 * (x_min + x_max) / (x_max - x_min),
		   -0.5 * (y_min + y_max) / (x_max - x_min),
		   0);
}

void FWElectronDetailView::fillData (const std::vector<DetId> &detids,
				   TEveCaloDataVec *data, 
				   double phi_seed)
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
	  // slice 1 is the non-seed clusters (which will show up in yellow)
	  int slice = 1;
	  if (find(seed_detids.begin(), seed_detids.end(), *k) != 
	      seed_detids.end()) {
	       // slice 0 is the seed cluster (which will show up in red)
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
	       double phi = v.Phi();
	       if (v.Phi() > phi_seed + M_PI)
		    phi -= 2 * M_PI;
	       if (v.Phi() < phi_seed - M_PI)
		    phi += 2 * M_PI;
	       data->AddTower(v.Eta() - 0.0174 / 2, v.Eta() + 0.0174 / 2, 
			      phi - 0.0174 / 2, phi + 0.0174 / 2);
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

