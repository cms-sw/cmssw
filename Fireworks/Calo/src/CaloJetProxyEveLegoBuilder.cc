// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetProxyEveLegoBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: CaloJetProxyEveLegoBuilder.cc,v 1.1 2008/03/20 09:39:25 dmytro Exp $
//

// system include files
#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveGeoShapeExtract.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"

// user include files
#include "Fireworks/Calo/interface/CaloJetProxyEveLegoBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloJetProxyEveLegoBuilder::CaloJetProxyEveLegoBuilder()
{
}

// CaloJetProxyEveLegoBuilder::CaloJetProxyEveLegoBuilder(const CaloJetProxyEveLegoBuilder& rhs)
// {
//    // do actual copying here;
// }

CaloJetProxyEveLegoBuilder::~CaloJetProxyEveLegoBuilder()
{
}

//
// assignment operators
//
// const CaloJetProxyEveLegoBuilder& CaloJetProxyEveLegoBuilder::operator=(const CaloJetProxyEveLegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   CaloJetProxyEveLegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }


void
CaloJetProxyEveLegoBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"JetsLego",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
   } else {
      tList->DestroyElements();
   }
   
   const reco::CaloJetCollection* jets=0;
   iItem->get(jets);
   if(0==jets) {
      std::cout <<"Failed to get CaloJets"<<std::endl;
      return;
   }
   
   fw::NamedCounter counter("jet");
   TColor* c = gROOT->GetColor( tList->GetMainColor() );
   Float_t rgba[4] = { 1, 0, 0, 1 };
   if (c) {
      rgba[0] = c->GetRed();
      rgba[1] = c->GetGreen();
      rgba[2] = c->GetBlue();
   }

   for(reco::CaloJetCollection::const_iterator jet = jets->begin(); 
       jet != jets->end(); ++jet, ++counter) {
      TEveElementList* container = new TEveElementList( counter.str().c_str() );
      
      TGeoTube *shape = new TGeoTube(0.48, 0.5, 0.0001);
      TEveTrans t;
      t.RotateLF(1,2,M_PI/2);
      t(1,4) = jet->eta(); 
      t(2,4) = jet->phi();
      t(3,4) = 0.1;
      TEveGeoShapeExtract *extract = new TEveGeoShapeExtract("outline");
      extract->SetTrans(t.Array());
      extract->SetRGBA(rgba);
      extract->SetRnrSelf(true);
      extract->SetRnrElements(true);
      extract->SetShape(shape);
      TEveElement* element = TEveGeoShape::ImportShapeExtract(extract, container);
      element->SetPickable(kTRUE);
      /* if ( jet->et()<15)
	element->SetMainTransparency(90);
      else
	element->SetMainTransparency(50);
       */
      tList->AddElement(container);
   }
}


