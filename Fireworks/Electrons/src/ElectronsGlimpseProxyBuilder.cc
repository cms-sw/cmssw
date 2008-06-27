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
// $Id: ElectronsGlimpseProxyBuilder.cc,v 1.1 2008/06/19 06:57:28 dmytro Exp $
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
#include "Fireworks/Electrons/interface/ElectronsGlimpseProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

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
   if(0==electrons) {
      std::cout <<"Failed to get GsfElectrons"<<std::endl;
      return;
   }
   
   fw::NamedCounter counter("electron");

   for(reco::GsfElectronCollection::const_iterator electron = electrons->begin(); 
       electron != electrons->end(); ++electron, ++counter) {

      TEveStraightLineSet* marker = new TEveStraightLineSet( counter.str().c_str() );
      marker->SetLineWidth(2);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      fw::addStraightLineSegment( marker, &*electron );
      tList->AddElement(marker);
   }
}

REGISTER_FWGLIMPSEDATAPROXYBUILDER(ElectronsGlimpseProxyBuilder,reco::GsfElectronCollection,"Electrons");

