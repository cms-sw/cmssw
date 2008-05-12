// -*- C++ -*-
//
// Package:     Calo
// Class  :     ElectronsProxySCBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: ElectronsProxySCBuilder.cc,v 1.11 2008/03/09 19:10:45 dmytro Exp $
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
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveSceneInfo.h"
#include "TEveViewer.h"
#include "TGLViewer.h"

// user include files
#include "Fireworks/Electrons/interface/ElectronsProxy3DBuilder.h"
#include "Fireworks/Core/interface/ElectronsProxySCBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

//
// constants, enums and typedefs
//
ElectronsProxySCBuilder *ElectronsProxySCBuilder::the_electron_sc_proxy = 0;
//
// static data member definitions
//

//
// constructors and destructor
//
ElectronsProxySCBuilder::ElectronsProxySCBuilder()
{
     // hack hack hack hack hack
     the_electron_sc_proxy = this;
}

// ElectronsProxySCBuilder::ElectronsProxySCBuilder(const ElectronsProxySCBuilder& rhs)
// {
//    // do actual copying here;
// }

ElectronsProxySCBuilder::~ElectronsProxySCBuilder()
{
   resetCenter();
}

//
// member functions
//
void ElectronsProxySCBuilder::build (TEveElementList **product) 
{
     // printf("calling ElectronsProxySCBuilder::buildRhoZ\n");
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
     using reco::GsfElectronCollection;
     const GsfElectronCollection *electrons = 0;
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
     int index=0;
     TEveRecTrack t;
//      t.fBeta = 1.;
     for(GsfElectronCollection::const_iterator i = electrons->begin();
	 i != electrons->end(); ++i, ++index) {
	  assert(i->gsfTrack().isNonnull());
#if 0
	  printf("GsfTrack contains %d inner states, %d outer states\n",
		 i->gsfTrack()->gsfExtra()->innerStateLocalParameters().size(),
		 i->gsfTrack()->gsfExtra()->outerStateLocalParameters().size());
	  std::vector<reco::GsfTrackExtra::LocalParameterVector> v_in = 
	       i->gsfTrack()->gsfExtra()->innerStateLocalParameters();
	  std::vector<reco::GsfTrackExtra::LocalParameterVector> v_out = 
	       i->gsfTrack()->gsfExtra()->outerStateLocalParameters();
	  for (int j = 0; j < std::max(v_out.size(), v_in.size()); ++j) {
	       if (j < v_in.size()) 
		    printf("v_in[%d]: %e\t%e\t%e\t%e\t%e\t\n", j, 
			   v_in[j][0],
			   v_in[j][1],
			   v_in[j][2],
			   v_in[j][3],
			   v_in[j][4]);
	       if (j < v_out.size()) 
		    printf("v_out[%d]: %e\t%e\t%e\t%e\t%e\t\n", j, 
			   v_out[j][0],
			   v_out[j][1],
			   v_out[j][2],
			   v_out[j][3],
			   v_out[j][4]);
	  }
#endif
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
	       TEveGeoShapeExtract* extract = m_item->getGeom()->getExtract(k->rawId(), /*corrected*/ true );
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
		    TColor* c = gROOT->GetColor(tList->GetMainColor());
		    if (c) {
			 rgba[0] = c->GetRed();
			 rgba[1] = c->GetGreen();
			 rgba[2] = c->GetBlue();
		    }
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
	  tList->AddElement(fw::getEcalCrystals(hits, *m_item->getGeom(), 
						sc.Eta(), sc.Phi()));
     }
}
