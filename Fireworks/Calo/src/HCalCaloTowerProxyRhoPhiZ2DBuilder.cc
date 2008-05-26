// -*- C++ -*-
//
// Package:     Calo
// Class  :     HCalCaloTowerProxyRhoPhiZ2DBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: HCalCaloTowerProxyRhoPhiZ2DBuilder.cc,v 1.2 2008/02/28 22:48:27 chrjones Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"

// user include files
#include "Fireworks/Calo/interface/HCalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDisplayEvent.h"

#include "Fireworks/Core/interface/FWRhoPhiZView.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HCalCaloTowerProxyRhoPhiZ2DBuilder::HCalCaloTowerProxyRhoPhiZ2DBuilder()
{
}

// HCalCaloTowerProxyRhoPhiZ2DBuilder::HCalCaloTowerProxyRhoPhiZ2DBuilder(const HCalCaloTowerProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

HCalCaloTowerProxyRhoPhiZ2DBuilder::~HCalCaloTowerProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void 
HCalCaloTowerProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"HCAL RhoPhi",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }
   
   const CaloTowerCollection* towers=0;
   iItem->get(towers);
   if(0==towers) {
      std::cout <<"Failed to get CaloTowers"<<std::endl;
      return;
   }
   tList->AddElement( TEveGeoShape::ImportShapeExtract( ECalCaloTowerProxyRhoPhiZ2DBuilder::getRhoPhiElements("towers", 
													      towers, 
													      iItem->defaultDisplayProperties().color(),
													      true,
													      1.5,
													      FWDisplayEvent::getCaloScale() ),
							0 ) );
}

void 
HCalCaloTowerProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"HCAL RhoZ",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }
   const CaloTowerCollection* towers=0;
   iItem->get(towers);
   if(0==towers) {
      std::cout <<"Failed to get CaloTowers"<<std::endl;
      return;
   }
   tList->AddElement( TEveGeoShape::ImportShapeExtract( ECalCaloTowerProxyRhoPhiZ2DBuilder::getRhoZElements("towers", 
													    towers, 
													    iItem->defaultDisplayProperties().color(),
													    true,
													    FWDisplayEvent::getCaloScale() ), 0 ) );
}

//
// const member functions
//

//
// static member functions
//
