// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1EtMissTrigProxyEveLegoBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: L1EtMissTrigProxyEveLegoBuilder.cc,v 1.2 2008/06/09 19:54:03 chrjones Exp $
//

// system include files
#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveGeoShapeExtract.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"

// user include files
#include "Fireworks/Calo/interface/L1EtMissTrigProxyEveLegoBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1EtMissTrigProxyEveLegoBuilder::L1EtMissTrigProxyEveLegoBuilder()
{


}

// L1EtMissTrigProxyEveLegoBuilder::L1EtMissTrigProxyEveLegoBuilder(const L1EtMissTrigProxyEveLegoBuilder& rhs)
// {
//    // do actual copying here;
// }

L1EtMissTrigProxyEveLegoBuilder::~L1EtMissTrigProxyEveLegoBuilder()
{
}

//
// assignment operators
//
// const L1EtMissTrigProxyEveLegoBuilder& L1EtMissTrigProxyEveLegoBuilder::operator=(const L1EtMissTrigProxyEveLegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   L1EtMissTrigProxyEveLegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }


void
L1EtMissTrigProxyEveLegoBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
  TEveElementList* tList = *product;

  // Make the eve element list
  if(0 == tList) {
    tList =  new TEveElementList(iItem->name().c_str(),"L1EtMissLego",true);
    *product = tList;
    tList->SetMainColor(iItem->defaultDisplayProperties().color());
  } else {
    tList->DestroyElements();
  }
   
  // Get the particle map collection for L1EtMissParticles
  l1extra::L1EtMissParticleCollection const * triggerColl=0;
  iItem->get(triggerColl);
  if(0==triggerColl) {
    std::cout <<"Failed to get L1EtMissTrig particle collection"<<std::endl;
    return;
  }
   
  // make a counter
  fw::NamedCounter counter("l1etMisstrigs");
  TColor* c = gROOT->GetColor( tList->GetMainColor() );
  Float_t rgba[4] = { 1, 0, 0, 1 };
  if (c) {
    rgba[0] = c->GetRed();
    rgba[1] = c->GetGreen();
    rgba[2] = c->GetBlue();
  }

  // Ready to loop over the triggered objects
  l1extra::L1EtMissParticleCollection::const_iterator trigIt = triggerColl->begin(),
    trigEnd = triggerColl->end();
  // Loop over triggered objects and make some 4-vectors
  for ( ; trigIt != trigEnd; ++trigIt ) {

    // Get eve container
    TEveElementList* container = new TEveElementList( counter.str().c_str() );
    TGeoTube *shape = new TGeoTube(0.48, 0.5, 0.0001);
    TEveTrans t;
    t.RotateLF(1,2,M_PI/2);

    // Find eta and phi based on object type
    double eta = trigIt->eta();
    double phi = trigIt->phi();

	
    // Fill eta and phi
    t(1,4) = eta;
    t(2,4) = phi;
    t(3,4) = 0.1;
    TEveGeoShapeExtract *extract = new TEveGeoShapeExtract("outline");
    extract->SetTrans(t.Array());
    extract->SetRGBA(rgba);
    extract->SetRnrSelf(true);
    extract->SetRnrElements(true);
    extract->SetShape(shape);
    TEveElement* element = TEveGeoShape::ImportShapeExtract(extract, container);
    element->SetPickable(kTRUE);
    /* if ( triggeredObjects[iTriggeredObjects]->p4().et()<15)
       element->SetMainTransparency(90);
       else
       element->SetMainTransparency(50);
    */
    tList->AddElement(container);
  }// end loop over em particle objects

  

}

REGISTER_FW3DLEGODATAPROXYBUILDER(L1EtMissTrigProxyEveLegoBuilder,l1extra::L1EtMissParticleCollection,"L1EtMissTrig");

