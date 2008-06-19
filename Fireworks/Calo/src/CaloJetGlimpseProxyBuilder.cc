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
// $Id: CaloJetGlimpseProxyBuilder.cc,v 1.4.2.1 2008/06/17 17:54:55 srappocc Exp $
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

// user include files
#include "Fireworks/Calo/interface/CaloJetGlimpseProxyBuilder.h"
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
   if(0==jets) {
      std::cout <<"Failed to get CaloJets"<<std::endl;
      return;
   }
   
   fw::NamedCounter counter("jet");

   for(reco::CaloJetCollection::const_iterator jet = jets->begin(); 
       jet != jets->end(); ++jet, ++counter) {

      TEveStraightLineSet* marker = new TEveStraightLineSet(counter.str().c_str());
      marker->SetLineWidth(1);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      fw::addStraightLineSegment( marker, &*jet );
      tList->AddElement(marker);
   }
}

REGISTER_FWGLIMPSEDATAPROXYBUILDER(CaloJetGlimpseProxyBuilder,reco::CaloJetCollection,"Jets");

