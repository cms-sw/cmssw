// -*- C++ -*-
//
// Package:     Calo
// Class  :     ElectronDetailView
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: ElectronDetailView.cc,v 1.6 2008/03/24 01:50:44 jmuelmen Exp $
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
#include "TEveText.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGTextView.h"

// user include files
#include "Fireworks/Electrons/interface/ElectronsProxy3DBuilder.h"
#include "Fireworks/Core/interface/ElectronDetailView.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
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
ElectronDetailView::ElectronDetailView()
{

}

// ElectronDetailView::ElectronDetailView(const ElectronDetailView& rhs)
// {
//    // do actual copying here;
// }

ElectronDetailView::~ElectronDetailView()
{
   resetCenter();
}

//
// member functions
//
void ElectronDetailView::build (TEveElementList **product, const FWModelId &id) 
{
     return build_projected(product, id);
}

void ElectronDetailView::build_3d (TEveElementList **product, const FWModelId &id) 
{
     m_item = id.item();
     // printf("calling ElectronDetailView::buildRhoZ\n");
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
     // get electrons
     resetCenter();
     using reco::PixelMatchGsfElectronCollection;
     const PixelMatchGsfElectronCollection *electrons = 0;
     // printf("getting electrons\n");
     m_item->get(electrons);
     // printf("got electrons\n");
     if (electrons == 0) {
	  std::cout <<"failed to get GSF electrons" << std::endl;
	  return;
     }
     // printf("%d GSF electrons\n", electrons->size());
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
     fwlite::Handle<EcalRecHitCollection> h_hits;
     const EcalRecHitCollection* hits(0);
     try {
	h_hits.getByLabel(*ev, "ecalRecHit", "EcalRecHitsEB");
	hits = h_hits.ptr();
     }
     catch (...) 
     {
	std::cout <<"no hits are ECAL rechits are available, show only crystal location" << std::endl;
     }
    
     TEveTrackPropagator *propagator = new TEveTrackPropagator();
     propagator->SetMagField( -4.0);
     propagator->SetMaxR( 180 );
     propagator->SetMaxZ( 430 );
     TEveRecTrack t;
     assert((unsigned int)id.index() < electrons->size());
//      t.fBeta = 1.;
     if (const reco::PixelMatchGsfElectron *i = &electrons->at(id.index())) {
	  assert(i->gsfTrack().isNonnull());
	  t.fP = TEveVector(i->gsfTrack()->px(),
			    i->gsfTrack()->py(),
			    i->gsfTrack()->pz());
	  t.fV = TEveVector(i->gsfTrack()->vx(),
			    i->gsfTrack()->vy(),
			    i->gsfTrack()->vz());
	  t.fSign = i->gsfTrack()->charge();
	  TEveTrack* trk = new TEveTrack(&t, propagator);
	  //const float rgba[4] = { 0, 1, 0, 1 };
// 	  trk->SetRGBA(rgba);
	  trk->SetLineColor((Color_t)kGreen);
	  trk->SetLineWidth(2);
	  TEvePathMark mark(TEvePathMark::kDaughter);
	  mark.fV = TEveVector(i->TrackPositionAtCalo().x(),
				i->TrackPositionAtCalo().y(),
				i->TrackPositionAtCalo().z());
	  trk->AddPathMark(mark);
	  trk->MakeTrack();
	  tList->AddElement(trk);
	  assert(i->superCluster().isNonnull());
	  TEveElementList* container = new TEveElementList("supercluster");
	  // figure out the extent of the supercluster
	  double min_phi = 100, max_phi = -100, min_eta = 100, max_eta = -100;
	  std::vector<DetId> detids = i->superCluster()->getHitsByDetId();
	  std::vector<DetId> seed_detids = i->superCluster()->seed()->
	       getHitsByDetId();
	  for (std::vector<DetId>::const_iterator k = detids.begin();
	       k != detids.end(); ++k) {
	       double size = 0.001; // default size
	       if ( hits ){
		  EcalRecHitCollection::const_iterator hit = hits->find(*k);
		  if (hit != hits->end()) 
		     size = hit->energy();
	       }
	       const TGeoHMatrix *matrix = m_item->getGeom()->getMatrix(k->rawId());
	       TEveGeoShapeExtract* extract = m_item->getGeom()->getExtract(k->rawId(), /*corrected*/ true  );
	       assert(extract != 0);
	       TEveTrans t = extract->GetTrans();
	       t.MoveLF(3, - size / 2);
	       // TGeoBBox *sc_box = new TGeoBBox(1.1, 1.1, size / 2, 0);
	       TGeoShape* crystal_shape = 0;
	       if ( const TGeoTrap* shape = dynamic_cast<const TGeoTrap*>(extract->GetShape()) ) {
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
// 		       printf("eta: %e\teta index: %d\t\tphi: %e\tphi index: %d\n",
// 			      v.Eta(), barrel_id.ieta(), v.Phi(), barrel_id.iphi());
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
	       if ( ! crystal_shape ) crystal_shape = new TGeoBBox(1.1, 1.1, size / 2, 0);
	       TEveGeoShapeExtract *extract2 = new TEveGeoShapeExtract("SC");
	       extract2->SetTrans(t.Array());
	       Float_t rgba[4] = { 1, 0, 0, 1 };
	       if (find(seed_detids.begin(), seed_detids.end(), *k) != 
		   seed_detids.end()) {
// 		    TColor* c = gROOT->GetColor(tList->GetMainColor());
// 		    if (c) {
// 			 rgba[0] = c->GetRed();
// 			 rgba[1] = c->GetGreen();
// 			 rgba[2] = c->GetBlue();
// 		    }
		    rgba[1] = 1;
	       } 
	       extract2->SetRGBA(rgba);
	       extract2->SetRnrSelf(true);
	       extract2->SetRnrElements(true);
	       extract2->SetShape(crystal_shape);
	       container->AddElement(TEveGeoShape::ImportShapeExtract(extract2,0));
/*
	       TGeoTrap *crystal = dynamic_cast<TGeoTrap *>(extract->GetShape());
	       assert(crystal != 0);
// 	       printf("%d\n", (char *)(&crystal->fH1) - (char *)crystal);
	       double *H1 = (double *)crystal + 30; // this is a kluge
	       printf("%f\n", *H1);
// 	       *H1++ = i->energy() / 10;
// 	       *H1++ = i->energy() / 10;
// 	       *H1++ = i->energy() / 10;
// 	       H1++;
// 	       *H1++ = i->energy() / 10;
// 	       *H1++ = i->energy() / 10;
// 	       *H1++ = i->energy() / 10;
	       TEveElement* shape = TEveGeoShape::ImportShapeExtract(extract,0);
	       shape->SetMainTransparency(50);
	       shape->SetMainColor(Color_t(kBlack + (int)floor(i->energy() + 10))); // tList->GetMainColor());
	       gEve->AddElement(shape);
	       tList->AddElement(shape);
*/
	  }
	  tList->AddElement(container);
	  TEvePointSet *trackpositionAtCalo = 
	       new TEvePointSet("sc trackpositionAtCalo", 1);
	  trackpositionAtCalo->SetNextPoint(i->TrackPositionAtCalo().x(),
					    i->TrackPositionAtCalo().y(),
					    i->TrackPositionAtCalo().z());
	  trackpositionAtCalo->SetMarkerStyle(20);
	  trackpositionAtCalo->SetMarkerSize(2);
	  trackpositionAtCalo->SetMarkerColor(kRed);
	  tList->AddElement(trackpositionAtCalo);
	  rotation_center[0] = i->TrackPositionAtCalo().x();
	  rotation_center[1] = i->TrackPositionAtCalo().y();
	  rotation_center[2] = i->TrackPositionAtCalo().z();
	  TEvePointSet *scposition = 
	       new TEvePointSet("sc position", 1);
	  scposition->SetNextPoint(i->caloPosition().x(),
				   i->caloPosition().y(),
				   i->caloPosition().z());
	  scposition->SetMarkerStyle(28);
	  scposition->SetMarkerSize(0.25);
	  scposition->SetMarkerColor(kBlue);
	  tList->AddElement(scposition);
	  TVector3 sc(i->caloPosition().x(),
		      i->caloPosition().y(),
		      i->caloPosition().z());
	  TVector3 v_pin_intersection;
	  v_pin_intersection.SetPtEtaPhi(
	       sc.Perp(),
	       sc.Eta() - i->deltaEtaSuperClusterTrackAtVtx(),
	       sc.Phi() - i->deltaPhiSuperClusterTrackAtVtx());
	  TEvePointSet *pinposition = 
	       new TEvePointSet("pin position", 1);
	  pinposition->SetNextPoint(v_pin_intersection.x(),
				    v_pin_intersection.y(),
				    v_pin_intersection.z());
	  pinposition->SetMarkerStyle(20);
	  pinposition->SetMarkerSize(2);
	  pinposition->SetMarkerColor(kCyan);
	  tList->AddElement(pinposition);
	  TEveElementList *all_crystals = 
	       fw::getEcalCrystals(hits, *m_item->getGeom(), sc.Eta(), sc.Phi());
	  all_crystals->SetMainColor((Color_t)kMagenta);
	  tList->AddElement(all_crystals);
     }
}

void ElectronDetailView::build_projected (TEveElementList **product, 
					  const FWModelId &id) 
{
     m_item = id.item();
     // printf("calling ElectronDetailView::buildRhoZ\n");
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
     // get electrons
     resetCenter();
     using reco::PixelMatchGsfElectronCollection;
     const PixelMatchGsfElectronCollection *electrons = 0;
     // printf("getting electrons\n");
     m_item->get(electrons);
     // printf("got electrons\n");
     if (electrons == 0) {
	  std::cout <<"failed to get GSF electrons" << std::endl;
	  return;
     }
     // printf("%d GSF electrons\n", electrons->size());
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
     fwlite::Handle<EcalRecHitCollection> h_hits;
     const EcalRecHitCollection* hits(0);
     try {
	h_hits.getByLabel(*ev, "ecalRecHit", "EcalRecHitsEB");
	hits = h_hits.ptr();
     }
     catch (...) 
     {
	std::cout <<"no hits are ECAL rechits are available, "
	     "show only crystal location" << std::endl;
     }
     const double scale = 100;
     float rgba[4] = { 1, 0, 0, 1 };
     if (const reco::PixelMatchGsfElectron *i = &electrons->at(id.index())) {
	  assert(i->gsfTrack().isNonnull());
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
	  for (std::vector<DetId>::const_iterator k = detids.begin();
	       k != detids.end(); ++k) {
	       double size = 50; // default size
	       if ( hits ){
		  EcalRecHitCollection::const_iterator hit = hits->find(*k);
		  if (hit != hits->end()) 
		     size = hit->energy();
	       }
	       const TGeoHMatrix *matrix = m_item->getGeom()->
		    getMatrix(k->rawId());
	       const TVector3 v(matrix->GetTranslation()[0], 
				matrix->GetTranslation()[1],
				matrix->GetTranslation()[2]);
	       if (k->subdetId() != EcalBarrel) 
		    continue;
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
	       t_box.SetPos(v.Eta() * scale,
			    v.Phi() * scale,
			    -0.11 - 0.1 * size);
//  	       t_box.MoveLF(1, v.Eta());
//  	       t_box.MoveLF(2, v.Phi());
// 	       t_box.Array()[12] = v.Eta();
// 	       t_box.Array()[13] = v.Phi();
// 	       for (int i = 0; i < 4; ++i) {
// 		    for (int j = 0; j < 4; ++j)
// 			 printf("%f ", t_box.Array()[4 * i + j]);
// 		    printf("\n");
// 	       }
	       TEveGeoShapeExtract *extract = new TEveGeoShapeExtract("ECAL crystal");
	       extract->SetShape(box);
	       extract->SetTrans(t_box.Array());
	       extract->SetRGBA(rgba);
	       container->AddElement(TEveGeoShape::ImportShapeExtract(extract, 0));
// 	       boxes->AddBox(v.Eta(), v.Phi(), 0, 0.1 * sqrt(size), 0.1 * sqrt(size), 1);
/*
	       TGeoTrap *crystal = dynamic_cast<TGeoTrap *>(extract->GetShape());
	       assert(crystal != 0);
// 	       printf("%d\n", (char *)(&crystal->fH1) - (char *)crystal);
	       double *H1 = (double *)crystal + 30; // this is a kluge
	       printf("%f\n", *H1);
// 	       *H1++ = i->energy() / 10;
// 	       *H1++ = i->energy() / 10;
// 	       *H1++ = i->energy() / 10;
// 	       H1++;
// 	       *H1++ = i->energy() / 10;
// 	       *H1++ = i->energy() / 10;
// 	       *H1++ = i->energy() / 10;
	       TEveElement* shape = TEveGeoShape::ImportShapeExtract(extract,0);
	       shape->SetMainTransparency(50);
	       shape->SetMainColor(Color_t(kBlack + (int)floor(i->energy() + 10))); // tList->GetMainColor());
	       gEve->AddElement(shape);
	       tList->AddElement(shape);
*/
	  }
	  container->AddElement(seed_boxes);
	  container->AddElement(non_seed_boxes);
	  tList->AddElement(container);
	  rotation_center[0] = i->superCluster()->position().eta() * scale;
	  rotation_center[1] = i->superCluster()->position().phi() * scale;
	  rotation_center[2] = 0;
// 	  rotation_center[0] = i->TrackPositionAtCalo().x();
// 	  rotation_center[1] = i->TrackPositionAtCalo().y();
// 	  rotation_center[2] = i->TrackPositionAtCalo().z();
	  TEvePointSet *scposition = 
	       new TEvePointSet("sc position", 1);
	  scposition->SetNextPoint(i->caloPosition().eta() * scale,
				   i->caloPosition().phi() * scale,
				   0);
	  scposition->SetMarkerStyle(28);
	  scposition->SetMarkerSize(0.25);
	  scposition->SetMarkerColor(kBlue);
	  tList->AddElement(scposition);
	  TEvePointSet *seedposition = 
	       new TEvePointSet("seed position", 1);
	  seedposition->SetNextPoint(i->superCluster()->seed()->position().eta() * scale,
				     i->superCluster()->seed()->position().phi() * scale,
				     0);
	  seedposition->SetMarkerStyle(28);
	  seedposition->SetMarkerSize(0.25);
	  seedposition->SetMarkerColor(kRed);
	  tList->AddElement(seedposition);
#if 0
	  TEvePointSet *trackpositionAtCalo = 
	       new TEvePointSet("sc trackpositionAtCalo", 1);
	  trackpositionAtCalo->SetNextPoint(i->TrackPositionAtCalo().eta() * scale,
					    i->TrackPositionAtCalo().phi() * scale,
					    0);
	  trackpositionAtCalo->SetMarkerStyle(20);
	  trackpositionAtCalo->SetMarkerSize(2);
	  trackpositionAtCalo->SetMarkerColor(kBlue);
	  tList->AddElement(trackpositionAtCalo);
	  TEvePointSet *pinposition = 
	       new TEvePointSet("pin position", 1);
	  pinposition->SetNextPoint((i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
				    (i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
				    0);
	  pinposition->SetMarkerStyle(20);
	  pinposition->SetMarkerSize(2);
	  pinposition->SetMarkerColor(kRed);
	  tList->AddElement(pinposition);
#else
	  TEveLine *trackpositionAtCalo = 
	       new TEveLine("sc trackpositionAtCalo");
	  trackpositionAtCalo->SetNextPoint(i->TrackPositionAtCalo().eta() * scale,
					    rotation_center[1] - 20,
					    0);
	  trackpositionAtCalo->SetNextPoint(i->TrackPositionAtCalo().eta() * scale,
					    rotation_center[1] + 20,
					    0);
	  trackpositionAtCalo->SetLineColor(kBlue);
	  tList->AddElement(trackpositionAtCalo);
	  trackpositionAtCalo = new TEveLine("sc trackpositionAtCalo");
	  trackpositionAtCalo->SetNextPoint(rotation_center[0] - 20,
					    i->TrackPositionAtCalo().phi() * scale,
					    0);
	  trackpositionAtCalo->SetNextPoint(rotation_center[0] + 20,
					    i->TrackPositionAtCalo().phi() * scale,
					    0);
	  trackpositionAtCalo->SetLineColor(kBlue);
	  tList->AddElement(trackpositionAtCalo);
	  TEveLine *pinposition = 
	       new TEveLine("pin position", 1);
	  pinposition->SetNextPoint((i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
				    rotation_center[1] - 20,
				    0);
	  pinposition->SetNextPoint((i->caloPosition().eta() - i->deltaEtaSuperClusterTrackAtVtx()) * scale,
				    rotation_center[1] + 20,
				    0);
	  pinposition->SetMarkerStyle(28);
	  pinposition->SetLineColor(kRed);
	  tList->AddElement(pinposition);
	  pinposition = new TEveLine("pin position", 1);
	  pinposition->SetNextPoint(rotation_center[0] - 20,
				    (i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
				    0);
	  pinposition->SetNextPoint(rotation_center[0] + 20,
				    (i->caloPosition().phi() - i->deltaPhiSuperClusterTrackAtVtx()) * scale,
				    0);
	  pinposition->SetMarkerStyle(28);
	  pinposition->SetLineColor(kRed);
	  tList->AddElement(pinposition);
#endif
	  // make labels
	  tList->AddElement(makeLabels(*i));
 	  TEveElementList *all_crystals = 
 	       getEcalCrystals(*m_item->getGeom(), 
 			       i->superCluster()->position().eta(), 
			       i->superCluster()->position().phi(),
			       5, 20);
 	  all_crystals->SetMainColor((Color_t)kMagenta);
 	  tList->AddElement(all_crystals);
     }
}

TEveElementList *ElectronDetailView::makeLabels (
     const reco::PixelMatchGsfElectron &electron) 
{
     TEveElementList *ret = new TEveElementList("electron labels");
#if DRAW_LABELS_IN_SEPARATE_VIEW
     // title
     text_view->AddLine("Electron detailed view"); 
     text_view->AddLine("");
     // summary
     if (electron.charge() > 0)
	  text_view->AddLine("charge = +1");
     else text_view->AddLine("charge = -1");
     char summary[128];
     sprintf(summary, "%s = %.1f GeV %10s = %.2f %10s = %.2f",
	     "ET", electron.caloEnergy() / cosh(electron.eta()), 
	     "eta", electron.eta(), 
	     "phi", electron.phi());
     text_view->AddLine(summary);
     // E/p, H/E
     char hoe[128];
     sprintf(hoe, "E/p = %.2f %13s = %.3f",
	     electron.eSuperClusterOverP(), 
	     "H/E", electron.hadronicOverEm());
     text_view->AddLine(hoe);
     // delta phi/eta in 
     char din[128];
     sprintf(din, "delta eta in = %.3f %16s = %.3f",
	     electron.deltaEtaSuperClusterTrackAtVtx(),
	     "delta phi in", electron.deltaPhiSuperClusterTrackAtVtx());
     text_view->AddLine(din);
     // delta phi/eta out 
     char dout[128];
     sprintf(dout, "delta eta out = %.3f %16s = %.3f",
	     electron.deltaEtaSeedClusterTrackAtCalo(),
	     "delta phi out", electron.deltaPhiSeedClusterTrackAtCalo());
     text_view->AddLine(dout);
     // legend
     text_view->AddLine("");
     text_view->AddLine("      red cross: track outer helix extrapolation");
     text_view->AddLine("     blue cross: track inner helix extrapolation");
     text_view->AddLine("      red point: seed cluster centroid");
     text_view->AddLine("     blue point: supercluster centroid");
     text_view->AddLine("   red crystals: seed cluster");
     text_view->AddLine("yellow crystals: other clusters");
#else
     // title
     TEveText* t = new TEveText("Electron detailed view");
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
	     electron.caloEnergy() / cosh(electron.eta()), 
	     electron.eta(), electron.phi());
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
	     electron.eSuperClusterOverP(), electron.hadronicOverEm());
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
	     electron.deltaEtaSuperClusterTrackAtVtx(),
	     electron.deltaPhiSuperClusterTrackAtVtx());
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
	     electron.deltaEtaSeedClusterTrackAtCalo(),
	     electron.deltaPhiSeedClusterTrackAtCalo());
     t = new TEveText(dout);
     t->PtrMainTrans()->MoveLF(1, rotation_center[0] + 3.0);
     t->PtrMainTrans()->MoveLF(2, rotation_center[1] + 10);
     t->SetMainColor((Color_t)(kWhite));
     t->SetFontSize(12);
     t->SetFontFile(8);
     t->SetLighting(kTRUE);
     ret->AddElement(t);
#endif
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
     return ret;
}

TEveElementList *ElectronDetailView::getEcalCrystals (const DetIdToMatrix &geo,
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
// 	       printf("pushing back (%d, %d)\n", i, j % 360);
	       if (EBDetId::validDetId(i, j % 360)) {
		    v.push_back(EBDetId(i, j % 360));
//  		    printf("pushing back (%d, %d)\n", i, j % 360);
	       }
	  }
     }
     return getEcalCrystals(geo, v);
}

TEveElementList *ElectronDetailView::getEcalCrystals (const DetIdToMatrix &geo,
						      const std::vector<DetId> &detids)
{
     TEveElementList *ret = new TEveElementList("Ecal crystals");
     for (std::vector<DetId>::const_iterator k = detids.begin();
	  k != detids.end(); ++k) {
	  const TGeoHMatrix *matrix = m_item->getGeom()->
	       getMatrix(k->rawId());
	  const TVector3 v(matrix->GetTranslation()[0], 
			   matrix->GetTranslation()[1],
			   matrix->GetTranslation()[2]);
// 	  printf("trying to add DetId %d... ", k->rawId());
	  if (k->subdetId() != EcalBarrel) {
// 	       printf("not in barrel\n");
	       continue;
	  }
// 	  printf("adding\n");
	  const double scale = 100;
	  float rgba[4] = { 1, 1, 0, 1 };
	  TGeoBBox *box = new TGeoBBox(0.48 * 0.0172 * scale, 
				       0.48 * 0.0172 * scale, 
				       0.01, 0);
	  TEveTrans t_box;
	  t_box.SetPos(v.Eta() * scale,
		       v.Phi() * scale,
		       -0.11);
	  TEveGeoShapeExtract *extract = new TEveGeoShapeExtract("ECAL crystal");
	  extract->SetShape(box);
	  extract->SetTrans(t_box.Array());
	  extract->SetRGBA(rgba);
	  TEveGeoShape *shape = TEveGeoShape::ImportShapeExtract(extract, 0);
 	  shape->SetMainTransparency(80);
	  ret->AddElement(shape);
     }
     return ret;
}
