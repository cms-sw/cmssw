// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWPhotonDetailView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWPhotonDetailView.cc,v 1.2 2009/01/23 21:35:46 amraktad Exp $
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
#include "TEveText.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGTextView.h"
#include "TEveBoxSet.h"

// user include files
#include "Fireworks/Core/interface/FWDetailView.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
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

class FWPhotonDetailView : public FWDetailView<reco::Photon> {

public:
   FWPhotonDetailView();
   virtual ~FWPhotonDetailView();

   virtual TEveElement* build (const FWModelId &id, const reco::Photon* );

protected:
   void setItem (const FWEventItem *iItem) {
      m_item = iItem;
   }
   void build_3d (TEveElementList **product, const FWModelId &id);
   TEveElement* build_projected (const FWModelId &id, const reco::Photon*);
   TEveElementList *makeLabels (const reco::Photon &);
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

private:
   FWPhotonDetailView(const FWPhotonDetailView&); // stop default
   const FWPhotonDetailView& operator=(const FWPhotonDetailView&); // stop default

   // ---------- member data --------------------------------
   const FWEventItem* m_item;

};

//
// constants, enums and typedefs
//
#define DRAW_LABELS_IN_SEPARATE_VIEW 1

//
// static data member definitions
//

//
// constructors and destructor
//
FWPhotonDetailView::FWPhotonDetailView()
{

}

// FWPhotonDetailView::FWPhotonDetailView(const FWPhotonDetailView& rhs)
// {
//    // do actual copying here;
// }

FWPhotonDetailView::~FWPhotonDetailView()
{
   resetCenter();
}

//
// member functions
//
TEveElement* FWPhotonDetailView::build (const FWModelId &id, const reco::Photon* iPhoton)
{
   return build_projected(id, iPhoton);
}

void FWPhotonDetailView::build_3d (TEveElementList **product, const FWModelId &id)
{
   m_item = id.item();
   // printf("calling FWPhotonDetailView::buildRhoZ\n");
   TEveElementList* tList = *product;
   if(0 == tList) {
      tList =  new TEveElementList(m_item->name().c_str(),"Supercluster RhoZ",true);
      *product = tList;
      tList->SetMainColor(m_item->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      return;
//        tList->DestroyElements();
   }
   // get photons
   resetCenter();
   using reco::PhotonCollection;
   const PhotonCollection *photons = 0;
   // printf("getting photons\n");
   m_item->get(photons);
   // printf("got photons\n");
   if (photons == 0) return;
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
   const EcalRecHitCollection* barrel_hits(0);
   try {
      h_barrel_hits.getByLabel(*ev, "ecalRecHit", "EcalRecHitsEB");
      barrel_hits = h_barrel_hits.ptr();
   }
   catch (...)
   {
      std::cout <<"no ECAL rechits are available, show only crystal location" << std::endl;
   }

   assert((unsigned int)id.index() < photons->size());
//      t.fBeta = 1.;
   if (const reco::Photon *i = &photons->at(id.index())) {
      assert(i->superCluster().isNonnull());
      TEveElementList* container = new TEveElementList("supercluster");
      // figure out the extent of the supercluster
      double min_phi = 100, max_phi = -100, min_eta = 100, max_eta = -100;
      std::vector<DetId> detids = i->superCluster()->getHitsByDetId();
      std::vector<DetId> seed_detids = i->superCluster()->seed()->
                                       getHitsByDetId();
      for (std::vector<DetId>::const_iterator k = detids.begin();
           k != detids.end(); ++k) {
         double size = 0.001;       // default size
         if ( barrel_hits ){
            EcalRecHitCollection::const_iterator hit = barrel_hits->find(*k);
            if (hit != barrel_hits->end())
               size = hit->energy();
         }
         const TGeoHMatrix *matrix = m_item->getGeom()->getMatrix(k->rawId());
         TEveGeoShape* egs = m_item->getGeom()->getShape(k->rawId(), /*corrected*/ true  );
         assert(egs != 0);
         TEveTrans &t = egs->RefMainTrans();
         t.MoveLF(3, -size / 2);
         TGeoShape* crystal_shape = 0;
         if ( const TGeoTrap* shape = dynamic_cast<const TGeoTrap*>(egs->GetShape()) ) {
            double scale = size/2/shape->GetDz();
            crystal_shape = new TGeoTrap( size/2,
                                          shape->GetTheta(), shape->GetPhi(),
                                          shape->GetH1()*scale + shape->GetH2()*(1-scale),
                                          shape->GetBl1()*scale + shape->GetBl2()*(1-scale),
                                          shape->GetTl1()*scale + shape->GetTl2()*(1-scale),
                                          shape->GetAlpha1(),
                                          shape->GetH2(), shape->GetBl2(), shape->GetTl2(),
                                          shape->GetAlpha2()
                                          );
            const TVector3 v(matrix->GetTranslation()[0],
                             matrix->GetTranslation()[1],
                             matrix->GetTranslation()[2]);
            if (k->subdetId() == EcalBarrel) {
               EBDetId barrel_id = *k;
               const double phi = v.Phi();
               const double eta = v.Eta();
//                     printf("eta: %e\teta index: %d\t\tphi: %e\tphi index: %d\n",
//                            v.Eta(), barrel_id.ieta(), v.Phi(), barrel_id.iphi());
               if (phi > max_phi)
                  max_phi = phi;
               if (phi < min_phi)
                  min_phi = phi;
               if (eta > max_eta)
                  max_eta = eta;
               if (eta < min_eta)
                  min_eta = eta;
            }
         }
         if ( !crystal_shape ) crystal_shape = new TGeoBBox(1.1, 1.1, size / 2, 0);
         egs->SetShape(crystal_shape);
         Float_t rgba[4] = { 1, 0, 0, 1 };
         if (find(seed_detids.begin(), seed_detids.end(), *k) !=
             seed_detids.end()) {
//                  TColor* c = gROOT->GetColor(tList->GetMainColor());
//                  if (c) {
//                       rgba[0] = c->GetRed();
//                       rgba[1] = c->GetGreen();
//                       rgba[2] = c->GetBlue();
//                  }
            rgba[1] = 1;
         }
         egs->SetMainColorRGB(rgba[0], rgba[1], rgba[2]);
         egs->SetRnrSelf(true);
         egs->SetRnrChildren(true);
         egs->SetShape(crystal_shape);
         container->AddElement(egs);
/*
               TGeoTrap *crystal = dynamic_cast<TGeoTrap *>(extract->GetShape());
               assert(crystal != 0);
   //          printf("%d\n", (char *)(&crystal->fH1) - (char *)crystal);
               double *H1 = (double *)crystal + 30; // this is a kluge
               printf("%f\n", *H1);
   //          *H1++ = i->energy() / 10;
   //          *H1++ = i->energy() / 10;
   //          *H1++ = i->energy() / 10;
   //          H1++;
   //          *H1++ = i->energy() / 10;
   //          *H1++ = i->energy() / 10;
   //          *H1++ = i->energy() / 10;
               TEveElement* shape = TEveGeoShape::ImportShapeExtract(extract,0);
               shape->SetMainTransparency(50);
               shape->SetMainColor(Color_t(kBlack + (int)floor(i->energy() + 10))); // tList->GetMainColor());
               gEve->AddElement(shape);
               tList->AddElement(shape);
 */
      }
      tList->AddElement(container);
      TEvePointSet *scposition =
         new TEvePointSet("sc position", 1);
      scposition->SetNextPoint(i->caloPosition().x(),
                               i->caloPosition().y(),
                               i->caloPosition().z());
      scposition->SetMarkerStyle(28);
      scposition->SetMarkerSize(0.25);
      scposition->SetMarkerColor(kBlue);
      tList->AddElement(scposition);
      rotationCenter()[0] = i->caloPosition().x();
      rotationCenter()[1] = i->caloPosition().y();
      rotationCenter()[2] = i->caloPosition().z();
      TVector3 sc(i->caloPosition().x(),
                  i->caloPosition().y(),
                  i->caloPosition().z());
      TEveElementList *all_crystals =
         fw::getEcalCrystals(barrel_hits, *m_item->getGeom(), sc.Eta(), sc.Phi());
      all_crystals->SetMainColor((Color_t)kMagenta);
      tList->AddElement(all_crystals);
   }
}

TEveElement* FWPhotonDetailView::build_projected (const FWModelId &id, const reco::Photon* iPhoton)
{
   if(0==iPhoton) {return 0;}
   m_item = id.item();
   // printf("calling FWPhotonDetailView::buildRhoZ\n");
   TEveElementList* tList =new TEveElementList(m_item->name().c_str(),"Supercluster RhoZ",true);
   tList->SetMainColor(m_item->defaultDisplayProperties().color());
   gEve->AddElement(tList);
   // get photons
   resetCenter();
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
   const EcalRecHitCollection* endcap_hits(0);
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
   if (const reco::Photon *i = iPhoton) {
      assert(i->superCluster().isNonnull());
      TEveElementList* container = new TEveElementList("supercluster");
      TEveElementList *seed_boxes =
         new TEveElementList("seed-cluster crystals");
      seed_boxes->SetMainColor((Color_t)kYellow);
      TEveElementList *non_seed_boxes =
         new TEveElementList("non-seed-cluster crystals");
      non_seed_boxes->SetMainColor((Color_t)kRed);
      TEveElementList *unclustered_boxes =
         new TEveElementList("unclustered crystals");
      unclustered_boxes->SetMainColor((Color_t)kMagenta);
      std::vector<DetId> detids = i->superCluster()->getHitsByDetId();
      std::vector<DetId> seed_detids = i->superCluster()->seed()->
                                       getHitsByDetId();
      const int subdetId =
         seed_detids.size() != 0 ? seed_detids.begin()->subdetId() : -1;
      const double scale = (subdetId == EcalBarrel) ? 100 : 1;
      for (std::vector<DetId>::const_iterator k = detids.begin();
           k != detids.end(); ++k) {
         double size = 50;       // default size
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
         TEveElementList *boxes = non_seed_boxes;
         rgba[0] = rgba[1] = 1; rgba[2] = 0;
         if (find(seed_detids.begin(), seed_detids.end(), *k) !=
             seed_detids.end()) {
            boxes = seed_boxes;
            rgba[0] = 1; rgba[1] = rgba[2] = 0;
         }
         TGeoBBox *box = new TGeoBBox(0.1 * sqrt(size),
                                      0.1 * sqrt(size),
                                      0.1 * size, 0);
         TEveTrans t_box;
         if (k->subdetId() == EcalBarrel) {
            t_box.SetPos(v.Eta() * scale,
                         v.Phi() * scale,
                         -0.11 - 0.1 * size);
         } else if (k->subdetId() == EcalEndcap) {
            t_box.SetPos(v.X() * scale,
                         v.Y() * scale,
                         -0.11 - 0.1 * size);
         }
         TEveGeoShape * ebox = new TEveGeoShape("ECAL crystal");
         ebox->SetShape(box);
         ebox->SetTransMatrix(t_box.Array());
         ebox->SetMainColorRGB(rgba[0], rgba[1], rgba[2]);
         container->AddElement(ebox);
      }
      container->AddElement(seed_boxes);
      container->AddElement(non_seed_boxes);
      tList->AddElement(container);
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
                                  0);
      } else if (subdetId == EcalEndcap) {
         scposition->SetNextPoint(i->caloPosition().x() * scale,
                                  i->caloPosition().y() * scale,
                                  0);
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
                                    0);
      } else if (subdetId == EcalEndcap) {
         seedposition->SetNextPoint(i->superCluster()->seed()->position().x() * scale,
                                    i->superCluster()->seed()->position().y() * scale,
                                    0);
      }
      seedposition->SetMarkerStyle(28);
      seedposition->SetMarkerSize(0.25);
      seedposition->SetMarkerColor(kRed);
      tList->AddElement(seedposition);
      // make labels
      tList->AddElement(makeLabels(*i));
      TEveElementList *all_crystals = 0;
      if (subdetId == EcalBarrel) {
         all_crystals = getEcalCrystalsBarrel(*m_item->getGeom(),
                                              i->superCluster()->position().eta(),
                                              i->superCluster()->position().phi(),
                                              5, 20);
      } else if (subdetId == EcalEndcap) {
         all_crystals = getEcalCrystalsEndcap(*m_item->getGeom(),
                                              i->superCluster()->position().x(),
                                              i->superCluster()->position().y(),
                                              i->superCluster()->position().z() > 0 ? 1 : -1,
                                              10, 10);
      }
      if (all_crystals != 0) {
         all_crystals->SetMainColor((Color_t)kMagenta);
         tList->AddElement(all_crystals);
      }
   }
   return tList;
}

TEveElementList *FWPhotonDetailView::makeLabels (const reco::Photon &photon)
{
   TEveElementList *ret = new TEveElementList("photon labels");
#if DRAW_LABELS_IN_SEPARATE_VIEW
   // summary
//      if (photon.charge() > 0)
//        textView()->AddLine("charge = +1");
//      else textView()->AddLine("charge = -1");
   char summary[128];
   sprintf(summary, "%s = %.1f GeV %10s = %.2f %10s = %.2f",
           "ET", photon.energy() / cosh(photon.eta()),
           "eta", photon.eta(),
           "phi", photon.phi());
   textView()->AddLine(summary);
   // E/p, H/E
   char hoe[128];
   sprintf(hoe, "%13s = %.3f",
           "H/E", photon.hadronicOverEm());
   textView()->AddLine(hoe);
   // delta phi/eta in
//      char din[128];
//      sprintf(din, "delta eta in = %.3f %16s = %.3f",
//           photon.deltaEtaSuperClusterTrackAtVtx(),
//           "delta phi in", photon.deltaPhiSuperClusterTrackAtVtx());
//      textView()->AddLine(din);
//      // delta phi/eta out
//      char dout[128];
//      sprintf(dout, "delta eta out = %.3f %16s = %.3f",
//           photon.deltaEtaSeedClusterTrackAtCalo(),
//           "delta phi out", photon.deltaPhiSeedClusterTrackAtCalo());
//      textView()->AddLine(dout);
   // legend
   textView()->AddLine("");
//      textView()->AddLine("      red cross: track outer helix extrapolation");
//      textView()->AddLine("     blue cross: track inner helix extrapolation");
   textView()->AddLine("      red point: seed cluster centroid");
   textView()->AddLine("     blue point: supercluster centroid");
   textView()->AddLine("   red crystals: seed cluster");
   textView()->AddLine("yellow crystals: other clusters");
#else
   // title
   TEveText* t = new TEveText("Photon detailed view");
   t->PtrMainTrans()->MoveLF(1, rotationCenter()[0] + 5);
   t->PtrMainTrans()->MoveLF(2, rotationCenter()[1] + 10);
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
   t->PtrMainTrans()->MoveLF(1, rotationCenter()[0] + 4.5);
   t->PtrMainTrans()->MoveLF(2, rotationCenter()[1] + 10);
   t->SetMainColor((Color_t)(kWhite));
   t->SetFontSize(12);
   t->SetFontFile(8);
   t->SetLighting(kTRUE);
   ret->AddElement(t);
   // E/p, H/E
   char hoe[128];
   sprintf(hoe, "H/E = %.3f",
           photon.hadronicOverEm());
   t = new TEveText(hoe);
   t->PtrMainTrans()->MoveLF(1, rotationCenter()[0] + 4.0);
   t->PtrMainTrans()->MoveLF(2, rotationCenter()[1] + 10);
   t->SetMainColor((Color_t)(kWhite));
   t->SetFontSize(12);
   t->SetFontFile(8);
   t->SetLighting(kTRUE);
   ret->AddElement(t);
   // delta phi/eta in
//      char din[128];
//      sprintf(din, "delta eta in = %.3f        delta phi in = %.3f",
//           photon.deltaEtaSuperClusterTrackAtVtx(),
//           photon.deltaPhiSuperClusterTrackAtVtx());
//      t = new TEveText(din);
//      t->PtrMainTrans()->MoveLF(1, rotationCenter()[0] + 3.5);
//      t->PtrMainTrans()->MoveLF(2, rotationCenter()[1] + 10);
//      t->SetMainColor((Color_t)(kWhite));
//      t->SetFontSize(12);
//      t->SetFontFile(8);
//      t->SetLighting(kTRUE);
//      ret->AddElement(t);
//      // delta phi/eta out
//      char dout[128];
//      sprintf(dout, "delta eta out = %.3f        delta phi out = %.3f",
//           photon.deltaEtaSeedClusterTrackAtCalo(),
//           photon.deltaPhiSeedClusterTrackAtCalo());
//      t = new TEveText(dout);
//      t->PtrMainTrans()->MoveLF(1, rotationCenter()[0] + 3.0);
//      t->PtrMainTrans()->MoveLF(2, rotationCenter()[1] + 10);
//      t->SetMainColor((Color_t)(kWhite));
//      t->SetFontSize(12);
//      t->SetFontFile(8);
//      t->SetLighting(kTRUE);
//      ret->AddElement(t);
#endif
   // eta, phi axis or x, y axis?
   assert(photon.superCluster().isNonnull());
   bool is_endcap = false;
   if (photon.superCluster()->getHitsByDetId().size() > 0 &&
       photon.superCluster()->getHitsByDetId().begin()->subdetId() == EcalEndcap)
      is_endcap = true;
   TEveLine *eta_line = new TEveLine;
   eta_line->SetNextPoint(rotationCenter()[0] - 15, rotationCenter()[1] - 40, 0);
   eta_line->SetNextPoint(rotationCenter()[0] - 10, rotationCenter()[1] - 40, 0);
   eta_line->SetLineColor((Color_t)kWhite);
   ret->AddElement(eta_line);
   TEveText *tt = new TEveText(is_endcap ? "x" : "eta");
   tt->PtrMainTrans()->MoveLF(1, rotationCenter()[0] - 9);
   tt->PtrMainTrans()->MoveLF(2, rotationCenter()[1] - 40);
   ret->AddElement(tt);
   TEveLine *phi_line = new TEveLine;
   phi_line->SetNextPoint(rotationCenter()[0] - 15, rotationCenter()[1] - 40, 0);
   phi_line->SetNextPoint(rotationCenter()[0] - 15, rotationCenter()[1] - 35, 0);
   phi_line->SetLineColor((Color_t)kWhite);
   ret->AddElement(phi_line);
   tt = new TEveText(is_endcap ? "y" : "phi");
   tt->PtrMainTrans()->MoveLF(1, rotationCenter()[0] - 15);
   tt->PtrMainTrans()->MoveLF(2, rotationCenter()[1] - 34);
   ret->AddElement(tt);
   return ret;
}

TEveElementList *FWPhotonDetailView::getEcalCrystalsBarrel (
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

TEveElementList *FWPhotonDetailView::getEcalCrystalsBarrel (
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

TEveElementList *FWPhotonDetailView::getEcalCrystalsEndcap (
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

TEveElementList *FWPhotonDetailView::getEcalCrystalsEndcap (
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

REGISTER_FWDETAILVIEW(FWPhotonDetailView);

