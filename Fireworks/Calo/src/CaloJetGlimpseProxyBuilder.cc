// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetGlimpseProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: CaloJetGlimpseProxyBuilder.cc,v 1.11 2008/11/04 20:29:24 amraktad Exp $
//

// system include files
#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"
#include "TEveStraightLineSet.h"
#include "TEveBoxSet.h"

// user include files
#include "Fireworks/Calo/interface/CaloJetGlimpseProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "Fireworks/Core/interface/FWGlimpseView.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Calo/interface/FWGlimpseEveJet.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloJetGlimpseProxyBuilder::CaloJetGlimpseProxyBuilder()
{
}

// CaloJetGlimpseProxyBuilder::CaloJetGlimpseProxyBuilder(const CaloJetGlimpseProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

CaloJetGlimpseProxyBuilder::~CaloJetGlimpseProxyBuilder()
{
}

//
// assignment operators
//
// const CaloJetGlimpseProxyBuilder& CaloJetGlimpseProxyBuilder::operator=(const CaloJetGlimpseProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   CaloJetGlimpseProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }


void
CaloJetGlimpseProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
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
   if(0==jets) return;

   fw::NamedCounter counter("jet");

   for(reco::CaloJetCollection::const_iterator jet = jets->begin();
       jet != jets->end(); ++jet, ++counter) {
      char title[1024];
      sprintf(title,"Jet %d, Et: %0.1f GeV",counter.index(),jet->et());
      FWGlimpseEveJet* cone = new FWGlimpseEveJet(&(*jet),counter.str().c_str(),title);
      cone->SetPickable(kTRUE);
      cone->SetMainColor(iItem->defaultDisplayProperties().color());
      cone->SetMainTransparency(50);
      cone->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
      cone->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
      cone->SetDrawConeCap(kFALSE);
      cone->SetMainTransparency(50);
      tList->AddElement(cone);
      scaler()->addElement(cone);
   }
}

REGISTER_FWGLIMPSEDATAPROXYBUILDER(CaloJetGlimpseProxyBuilder,reco::CaloJetCollection,"Jets");

