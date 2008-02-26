// -*- C++ -*-
//
// Package:     Calo
// Class  :     ElectronsProxyRhoPhiZ2DBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: ElectronsProxyRhoPhiZ2DBuilder.cc,v 1.3 2008/02/15 06:41:20 jmuelmen Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"

// user include files
#include "Fireworks/Electrons/interface/ElectronsProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ElectronsProxyRhoPhiZ2DBuilder::ElectronsProxyRhoPhiZ2DBuilder()
{
}

// ElectronsProxyRhoPhiZ2DBuilder::ElectronsProxyRhoPhiZ2DBuilder(const ElectronsProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

ElectronsProxyRhoPhiZ2DBuilder::~ElectronsProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void 
ElectronsProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
     TEveElementList* tList = *product;

     // printf("calling ElectronsProxyRhoPhiZ2DBuilder::buildRhiPhi\n");
     if(0 == tList) {
	  tList =  new TEveElementList(iItem->name().c_str(),"Electron RhoPhi",true);
	  *product = tList;
	  tList->SetMainColor(iItem->defaultDisplayProperties().color());
	  gEve->AddElement(tList);
     } else {
	  tList->DestroyElements();
     }
     // get electrons
     using reco::PixelMatchGsfElectronCollection;
     const PixelMatchGsfElectronCollection *electrons = 0;
     // printf("getting electrons\n");
     iItem->get(electrons);
     // printf("got electrons\n");
     if (electrons == 0) {
	  std::cout <<"failed to get GSF electrons" << std::endl;
	  return;
     }
     // printf("%d GSF electrons\n", electrons->size());
     // loop over electrons
     using std::string;
     string name = "superclusters";
     TEveGeoShapeExtract* container = new TEveGeoShapeExtract(name.c_str());
     char index[3] = "00";
     for (PixelMatchGsfElectronCollection::const_iterator i = electrons->begin();
	  i != electrons->end(); ++i, ++index[0]) {
	  assert(i->superCluster().isNonnull());
// 	  const SuperCluster &sc = i->superCluster().product();
	  DetId id = i->superCluster()->seed()->getHitsByDetId()[0];
	  if (id.subdetId() != EcalBarrel) 
	       // skip these for now
	       continue;
#if 0
	  double size = 1;
	  double r = 122;
	  double phi = i->superCluster()->position().phi();
	  double phi_deg = phi * 180 / M_PI;
	  TGeoBBox *sc_box = new TGeoTubeSeg(r - 1, r + 1, 1, 
					     phi_deg - 15, phi_deg + 15);
	  TEveGeoShapeExtract *extract = new TEveGeoShapeExtract((name + index).c_str());
	  TColor* c = gROOT->GetColor(tList->GetMainColor());
	  Float_t rgba[4] = { 1, 0, 0, 1 };
	  if (c) {
	       rgba[0] = c->GetRed();
	       rgba[1] = c->GetGreen();
	       rgba[2] = c->GetBlue();
	  }
	  extract->SetRGBA(rgba);
	  extract->SetRnrSelf(true);
	  extract->SetRnrElements(true);
	  extract->SetShape(sc_box);
	  container->AddElement(extract);
#else
	  std::vector<DetId> detids = i->superCluster()->getHitsByDetId();
	  for (std::vector<DetId>::const_iterator k = detids.begin();
	       k != detids.end(); ++k, ++index[1]) {
// 	       const TGeoHMatrix* matrix = m_item->getGeom()->getMatrix( k->rawId() );
	       TEveGeoShapeExtract* extract = m_item->getGeom()->getExtract( k->rawId() );
	       assert(extract != 0);
	       TVector3 v(extract->GetTrans()[12], 
			  extract->GetTrans()[13], 
			  extract->GetTrans()[14]);
	       TEveElement* shape = TEveGeoShape::ImportShapeExtract(extract,0);
	       shape->SetMainTransparency(50);
	       shape->SetMainColor(tList->GetMainColor());
// 		    tList->AddElement(shape);
	       // double size = 1;
	       double r = 122;
	       double phi = v.Phi();
	       double phi_deg_min = (phi - 0.0085) * 180 / M_PI;
	       double phi_deg_max = (phi + 0.0085) * 180 / M_PI;
	       TGeoBBox *sc_box = new TGeoTubeSeg(r - 1, r + 1, 1, 
						  phi_deg_min, phi_deg_max);
	       TEveGeoShapeExtract *extract2 = new TEveGeoShapeExtract((name + index).c_str());
	       TColor* c = gROOT->GetColor(tList->GetMainColor());
	       Float_t rgba[4] = { 1, 0, 0, 1 };
	       if (c) {
		    rgba[0] = c->GetRed();
		    rgba[1] = c->GetGreen();
		    rgba[2] = c->GetBlue();
	       }
	       extract2->SetRGBA(rgba);
	       extract2->SetRnrSelf(true);
	       extract2->SetRnrElements(true);
	       extract2->SetShape(sc_box);
	       container->AddElement(extract2);
	  }
#endif
     }
     tList->AddElement(TEveGeoShape::ImportShapeExtract(container, 0));
}

void 
ElectronsProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					  TEveElementList** product)
{
     // printf("calling ElectronsProxyRhoPhiZ2DBuilder::buildRhoZ\n");
     TEveElementList* tList = *product;
     if(0 == tList) {
	  tList =  new TEveElementList(iItem->name().c_str(),"Electron RhoZ",true);
	  *product = tList;
	  tList->SetMainColor(iItem->defaultDisplayProperties().color());
	  gEve->AddElement(tList);
     } else {
	  tList->DestroyElements();
     }
     // get electrons
     using reco::PixelMatchGsfElectronCollection;
     const PixelMatchGsfElectronCollection *electrons = 0;
     // printf("getting electrons\n");
     iItem->get(electrons);
     // printf("got electrons\n");
     if (electrons == 0) {
	  std::cout <<"failed to get GSF electrons" << std::endl;
	  return;
     }
     // printf("%d GSF electrons\n", electrons->size());
     // loop over electrons
     using std::string;
     string name = "superclusters";
     TEveGeoShapeExtract* container = new TEveGeoShapeExtract(name.c_str());
     char index[3] = "00";
     for (PixelMatchGsfElectronCollection::const_iterator i = electrons->begin();
	  i != electrons->end(); ++i, ++index[0]) {
	  assert(i->superCluster().isNonnull());
// 	  const SuperCluster &sc = i->superCluster().product();
	  DetId id = i->superCluster()->seed()->getHitsByDetId()[0];
	  if (id.subdetId() != EcalBarrel) 
	       // skip these for now
	       continue;
#if 0
	  double size = 1;
	  TGeoBBox *sc_box = new TGeoBBox(1, size, 15, 0);
	  double r = 122;
	  TEveTrans t;
	  t(1,1) = 1; t(1,2) = 0; t(1,3) = 0;
	  t(2,1) = 0; t(2,2) = 1; t(2,3) = 0;
	  t(3,1) = 0; t(3,2) = 0; t(3,3) = 1;
	  t(1,4) = 0; 
 	  t(2,4) = (r + size) * (i->superCluster()->position().y() > 0 ? 1 : -1); 
	  t(3,4) = i->superCluster()->position().z();
// 	  t.RotateLF(2,3,-M_PI/2);
// 	  t.RotateLF(1,2,M_PI/2);
// 	  if ( i->superCluster()->position().phi() < M_PI ) 
// 	       t.RotatePF(2,3,M_PI/2-theta);
// 	  else {
// 	       t.RotatePF(1,2,M_PI);
// 	       t.RotatePF(2,3,-M_PI/2+theta);
// 	  }
	  TEveGeoShapeExtract *extract = new TEveGeoShapeExtract((name + index).c_str());
	  extract->SetTrans(t.Array());
	  TColor* c = gROOT->GetColor(tList->GetMainColor());
	  Float_t rgba[4] = { 1, 0, 0, 1 };
	  if (c) {
	       rgba[0] = c->GetRed();
	       rgba[1] = c->GetGreen();
	       rgba[2] = c->GetBlue();
	  }
	  extract->SetRGBA(rgba);
	  extract->SetRnrSelf(true);
	  extract->SetRnrElements(true);
	  extract->SetShape(sc_box);
	  container->AddElement(extract);
#else
	  std::vector<DetId> detids = i->superCluster()->getHitsByDetId();
	  for (std::vector<DetId>::const_iterator k = detids.begin();
	       k != detids.end(); ++k, ++index[1]) {
// 	       const TGeoHMatrix* matrix = m_item->getGeom()->getMatrix( k->rawId() );
	       TEveGeoShapeExtract* extract = m_item->getGeom()->getExtract( k->rawId() );
	       if(0!=extract) {
 		    // TEveElement* shape = TEveGeoShape::ImportShapeExtract(extract,0);
//  		    shape->SetMainTransparency(100);
//  		    shape->SetMainColor(0);
// 		    tList->AddElement(shape);
		    double size = 1;
		    TGeoBBox *sc_box = new TGeoBBox(1, size, 1.1, 0);
		    double r_draw = 122;
		    double r_crystal = 
			 sqrt(extract->GetTrans()[12] * extract->GetTrans()[12] +
			      extract->GetTrans()[13] * extract->GetTrans()[13]);
		    double z = extract->GetTrans()[14] * r_draw / r_crystal;
		    TEveTrans t;
		    t(1,1) = 1; t(1,2) = 0; t(1,3) = 0;
		    t(2,1) = 0; t(2,2) = 1; t(2,3) = 0;
		    t(3,1) = 0; t(3,2) = 0; t(3,3) = 1;
		    t(1,4) = 0; 
		    t(2,4) = (r_draw + size) * 
			 (i->superCluster()->position().y() > 0 ? 1 : -1); 
		    t(3,4) = z;
		    TEveGeoShapeExtract *extract2 = new 
			 TEveGeoShapeExtract((name + index).c_str());
		    extract2->SetTrans(t.Array());
		    TColor* c = gROOT->GetColor(tList->GetMainColor());
		    Float_t rgba[4] = { 1, 0, 0, 1 };
		    if (c) {
			 rgba[0] = c->GetRed();
			 rgba[1] = c->GetGreen();
			 rgba[2] = c->GetBlue();
		    }
		    extract2->SetRGBA(rgba);
		    extract2->SetRnrSelf(true);
		    extract2->SetRnrElements(true);
		    extract2->SetShape(sc_box);
		    container->AddElement(extract2);
// 		    delete extract;
	       }
	  }
#endif
     }
     tList->AddElement(TEveGeoShape::ImportShapeExtract(container, 0));
#if 0
	  // use sigma_eta_eta as the width of the SC representation
	  // printf("getting cluster shapes\n");
	  using reco::BasicClusterShapeAssociationCollection;
	  const BasicClusterShapeAssociationCollection *shapes = 0;
	  iItem->get(shapes);
	  // printf("got cluster shapes\n");
	  for (BasicClusterShapeAssociationCollection::const_iterator j
		    = shapes->begin(); j != shapes->end(); ++j) {
	       // Get the ClusterShapeRef corresponding to the BasicCluster
	       using reco::ClusterShapeRef;
	       const ClusterShapeRef &seedShapeRef = j->val;
	       double see = sqrt(seedShapeRef->covEtaEta());
	       // printf("see = %f\n", see);
	  }
     }
#endif
}
