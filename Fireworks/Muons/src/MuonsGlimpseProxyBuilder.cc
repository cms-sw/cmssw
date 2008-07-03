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
// $Id: MuonsGlimpseProxyBuilder.cc,v 1.3 2008/06/28 22:15:54 dmytro Exp $
//

// system include files
#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveGeoShapeExtract.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"
#include "TEveStraightLineSet.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Muons/interface/MuonsGlimpseProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWGlimpseView.h"

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
   if(0==muons) {
      std::cout <<"Failed to get Muonss"<<std::endl;
      return;
   }
   
   fw::NamedCounter counter("muon");

   for(reco::MuonCollection::const_iterator muon = muons->begin(); 
       muon != muons->end(); ++muon, ++counter) {
      char title[1024];
      sprintf(title,"Muon %d, Pt: %0.1f GeV",counter.index(),muon->pt());
      TEveStraightLineSet* marker = new TEveStraightLineSet(counter.str().c_str(),title);
      marker->SetLineWidth(2);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      fw::addStraightLineSegment( marker, &*muon, FWGlimpseView::getScale() );
      tList->AddElement(marker);
   }
}

REGISTER_FWGLIMPSEDATAPROXYBUILDER(MuonsGlimpseProxyBuilder,reco::MuonCollection,"Muons");

