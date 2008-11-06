// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1MuonTrigProxyEveLegoBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: L1MuonTrigProxyEveLegoBuilder.cc,v 1.4 2008/11/06 19:49:22 amraktad Exp $
//

// system include files
#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"

// user include files
#include "Fireworks/Calo/interface/L1MuonTrigProxyEveLegoBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1MuonTrigProxyEveLegoBuilder::L1MuonTrigProxyEveLegoBuilder()
{


}

// L1MuonTrigProxyEveLegoBuilder::L1MuonTrigProxyEveLegoBuilder(const L1MuonTrigProxyEveLegoBuilder& rhs)
// {
//    // do actual copying here;
// }

L1MuonTrigProxyEveLegoBuilder::~L1MuonTrigProxyEveLegoBuilder()
{
}

//
// assignment operators
//
// const L1MuonTrigProxyEveLegoBuilder& L1MuonTrigProxyEveLegoBuilder::operator=(const L1MuonTrigProxyEveLegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   L1MuonTrigProxyEveLegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }


void
L1MuonTrigProxyEveLegoBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
  TEveElementList* tList = *product;
  TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());

  // Make the eve element list
  if(0 == tList) {
    tList =  new TEveElementList(iItem->name().c_str(),"L1MuonLego",true);
    *product = tList;
    tList->SetMainColor(iItem->defaultDisplayProperties().color());
  } else {
    tList->DestroyElements();
  }
   
  // Get the particle map collection for L1MuonParticles
  l1extra::L1MuonParticleCollection const * triggerColl=0;
  iItem->get(triggerColl);
  if(0==triggerColl) return;
   
  // make a counter
  fw::NamedCounter counter("l1muontrigs");
  TColor* c = gROOT->GetColor( tList->GetMainColor() );
  Float_t rgba[4] = { 1, 0, 0, 1 };
  if (c) {
    rgba[0] = c->GetRed();
    rgba[1] = c->GetGreen();
    rgba[2] = c->GetBlue();
  }

  // Ready to loop over the triggered objects
  l1extra::L1MuonParticleCollection::const_iterator trigIt = triggerColl->begin(),
    trigEnd = triggerColl->end();
  // Loop over triggered objects and make some 4-vectors
  TGeoTube *shape = new TGeoTube(0.48, 0.5, 0.0001);
  for ( ; trigIt != trigEnd; ++trigIt ) {

    // Get eve container
    TEveElementList* container = new TEveElementList( counter.str().c_str() );
    TEveTrans t;
    t.RotateLF(1,2,M_PI/2);

    // Find eta and phi based on object type
    double eta = trigIt->eta();
    double phi = trigIt->phi();

	
    // Fill eta and phi
    t(1,4) = eta;
    t(2,4) = phi;
    t(3,4) = 0.1;
    TEveGeoShape *egs = new TEveGeoShape("outline");
    egs->SetTransMatrix(t.Array());
    egs->SetMainColorRGB(rgba[0], rgba[1], rgba[2]);
    egs->SetShape(shape);
    egs->SetPickable(kTRUE);
    container->AddElement(egs);
    /* if ( triggeredObjects[iTriggeredObjects]->p4().et()<15)
       element->SetMainTransparency(90);
       else
       element->SetMainTransparency(50);
    */
    tList->AddElement(container);
  }// end loop over em particle objects

  

}

REGISTER_FW3DLEGODATAPROXYBUILDER(L1MuonTrigProxyEveLegoBuilder,l1extra::L1MuonParticleCollection,"L1MuonTrig");

