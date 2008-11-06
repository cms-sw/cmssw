// -*- C++ -*-
//
// Package:     Electron
// Class  :     PatElectronsGlimpseProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: PatElectronsGlimpseProxyBuilder.cc,v 1.2 2008/11/04 20:29:25 amraktad Exp $
//

// system include files
#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Electrons/interface/PatElectronsGlimpseProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "Fireworks/Core/interface/FWGlimpseView.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PatElectronsGlimpseProxyBuilder::PatElectronsGlimpseProxyBuilder()
{
}

// PatElectronsGlimpseProxyBuilder::PatElectronsGlimpseProxyBuilder(const PatElectronsGlimpseProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

PatElectronsGlimpseProxyBuilder::~PatElectronsGlimpseProxyBuilder()
{
}

void
PatElectronsGlimpseProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"ElectronsLego",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
   } else {
      tList->DestroyElements();
   }

   const std::vector<pat::Electron>* electrons=0;
   iItem->get(electrons);
   if(0==electrons) return;

   fw::NamedCounter counter("electron");

   for(std::vector<pat::Electron>::const_iterator electron = electrons->begin();
       electron != electrons->end(); ++electron, ++counter) {
      char title[1024];
      snprintf(title,1024,"Electron %d, Pt: %0.1f GeV",counter.index(), electron->pt());
      FWEveScalableStraightLineSet* marker = new FWEveScalableStraightLineSet( counter.str().c_str(), title );
      marker->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
      marker->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
      marker->SetLineWidth(2);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      fw::addStraightLineSegment( marker, &*electron, 1.0 );
      tList->AddElement(marker);
      //add to scaler at end so that it can scale the line after all ends have been added
      scaler()->addElement(marker);
   }
}

REGISTER_FWGLIMPSEDATAPROXYBUILDER(PatElectronsGlimpseProxyBuilder,std::vector<pat::Electron>,"PatElectrons");

