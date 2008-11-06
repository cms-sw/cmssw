// -*- C++ -*-
//
// Package:     Calo
// Class  :     PatJetGlimpseProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: PatJetGlimpseProxyBuilder.cc,v 1.2 2008/11/04 20:29:25 amraktad Exp $
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
#include "Fireworks/Calo/interface/PatJetGlimpseProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
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
PatJetGlimpseProxyBuilder::PatJetGlimpseProxyBuilder()
{
}


PatJetGlimpseProxyBuilder::~PatJetGlimpseProxyBuilder()
{
}

void
PatJetGlimpseProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"GlimpseJets",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
   } else {
      tList->DestroyElements();
   }

   const std::vector<pat::Jet>* jets=0;
   iItem->get(jets);
   if(0==jets) return;

   fw::NamedCounter counter("jet");

   for(std::vector<pat::Jet>::const_iterator jet = jets->begin();
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

REGISTER_FWGLIMPSEDATAPROXYBUILDER(PatJetGlimpseProxyBuilder,std::vector<pat::Jet>,"PatJets");

