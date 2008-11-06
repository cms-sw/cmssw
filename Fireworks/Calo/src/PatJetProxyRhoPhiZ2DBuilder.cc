// -*- C++ -*-
//
// Package:     Calo
// Class  :     PatJetProxyRhoPhiZ2DBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: PatJetProxyRhoPhiZ2DBuilder.cc,v 1.2 2008/11/04 20:29:25 amraktad Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEvePointSet.h"
#include "TEveScalableStraightLineSet.h"
#include "TEveCompound.h"
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

// user include files
#include "Fireworks/Calo/interface/CaloJetProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Calo/interface/PatJetProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
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
PatJetProxyRhoPhiZ2DBuilder::PatJetProxyRhoPhiZ2DBuilder()
{
}

// PatJetProxyRhoPhiZ2DBuilder::PatJetProxyRhoPhiZ2DBuilder(const PatJetProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

PatJetProxyRhoPhiZ2DBuilder::~PatJetProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void
PatJetProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Jets RhoPhi",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const std::vector<pat::Jet>* jets=0;
   iItem->get(jets);
   if(0==jets) return;

   fw::NamedCounter counter("jet");

   for(std::vector<pat::Jet>::const_iterator jet = jets->begin();
       jet != jets->end(); ++jet, ++counter) {
      CaloJetProxyRhoPhiZ2DBuilder::buildJetRhoPhi( iItem, &*jet, tList, counter );
   }
}

void
PatJetProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;
   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Jets RhoZ",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const std::vector<pat::Jet>* jets=0;
   iItem->get(jets);
   if(0==jets) return;

   fw::NamedCounter counter("jet");

   for(std::vector<pat::Jet>::const_iterator jet = jets->begin();
       jet != jets->end(); ++jet, ++counter) {
      CaloJetProxyRhoPhiZ2DBuilder::buildJetRhoZ( iItem, &*jet, tList, counter );
   }
}

REGISTER_FWRPZ2DDATAPROXYBUILDER(PatJetProxyRhoPhiZ2DBuilder,std::vector<pat::Jet>,"PatJets");

