// -*- C++ -*-
//
// Package:     Calo
// Class  :     MuonsGlimpseProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: MuonsGlimpseProxyBuilder.cc,v 1.8 2008/11/04 20:29:26 amraktad Exp $
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
#include "Fireworks/Muons/interface/MuonsGlimpseProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWGlimpseView.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonsGlimpseProxyBuilder::MuonsGlimpseProxyBuilder()
{
}

// MuonsGlimpseProxyBuilder::MuonsGlimpseProxyBuilder(const MuonsGlimpseProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

MuonsGlimpseProxyBuilder::~MuonsGlimpseProxyBuilder()
{
}

void
MuonsGlimpseProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Muons",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
   } else {
      tList->DestroyElements();
   }

   const reco::MuonCollection* muons=0;
   iItem->get(muons);
   if(0==muons) return;

   fw::NamedCounter counter("muon");

   for(reco::MuonCollection::const_iterator muon = muons->begin();
       muon != muons->end(); ++muon, ++counter) {
      char title[1024];
      sprintf(title,"Muon %d, Pt: %0.1f GeV",counter.index(),muon->pt());
      FWEveScalableStraightLineSet* marker = new FWEveScalableStraightLineSet( counter.str().c_str(), title );
      marker->SetLineWidth(2);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      marker->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
      marker->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
      fw::addStraightLineSegment( marker, &*muon, 1.0 );
      tList->AddElement(marker);
      //add to scaler at end so that it can scale the line after all ends have been added
      scaler()->addElement(marker);
   }
}

REGISTER_FWGLIMPSEDATAPROXYBUILDER(MuonsGlimpseProxyBuilder,reco::MuonCollection,"Muons");

