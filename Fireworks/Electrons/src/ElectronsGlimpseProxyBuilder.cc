// -*- C++ -*-
//
// Package:     Electron
// Class  :     ElectronsGlimpseProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: ElectronsGlimpseProxyBuilder.cc,v 1.8 2008/11/04 20:29:25 amraktad Exp $
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
#include "Fireworks/Electrons/interface/ElectronsGlimpseProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
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
ElectronsGlimpseProxyBuilder::ElectronsGlimpseProxyBuilder()
{
}

// ElectronsGlimpseProxyBuilder::ElectronsGlimpseProxyBuilder(const ElectronsGlimpseProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

ElectronsGlimpseProxyBuilder::~ElectronsGlimpseProxyBuilder()
{
}

void
ElectronsGlimpseProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"ElectronsLego",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
   } else {
      tList->DestroyElements();
   }

   const reco::GsfElectronCollection* electrons=0;
   iItem->get(electrons);
   if(0==electrons) return;

   fw::NamedCounter counter("electron");

   for(reco::GsfElectronCollection::const_iterator electron = electrons->begin();
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

REGISTER_FWGLIMPSEDATAPROXYBUILDER(ElectronsGlimpseProxyBuilder,reco::GsfElectronCollection,"Electrons");

