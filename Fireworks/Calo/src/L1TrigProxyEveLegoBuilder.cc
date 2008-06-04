// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1TrigProxyEveLegoBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: L1TrigProxyEveLegoBuilder.cc,v 1.2 2008/05/12 15:38:00 dmytro Exp $
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
#include "Fireworks/Calo/interface/L1TrigProxyEveLegoBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMapFwd.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TrigProxyEveLegoBuilder::L1TrigProxyEveLegoBuilder()
{


}

// L1TrigProxyEveLegoBuilder::L1TrigProxyEveLegoBuilder(const L1TrigProxyEveLegoBuilder& rhs)
// {
//    // do actual copying here;
// }

L1TrigProxyEveLegoBuilder::~L1TrigProxyEveLegoBuilder()
{
}

//
// assignment operators
//
// const L1TrigProxyEveLegoBuilder& L1TrigProxyEveLegoBuilder::operator=(const L1TrigProxyEveLegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   L1TrigProxyEveLegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }


void
L1TrigProxyEveLegoBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
  TEveElementList* tList = *product;

  // Make the eve element list
  if(0 == tList) {
    tList =  new TEveElementList(iItem->name().c_str(),"L1Lego",true);
    *product = tList;
    tList->SetMainColor(iItem->defaultDisplayProperties().color());
  } else {
    tList->DestroyElements();
  }
   
  // Get the particle map collection for L1
  l1extra::L1ParticleMapCollection const * particleMaps=0;
  iItem->get(particleMaps);
  if(0==particleMaps) {
    std::cout <<"Failed to get L1Trig particle map"<<std::endl;
    return;
  }
   
  // make a counter
  fw::NamedCounter counter("l1trigs");
  TColor* c = gROOT->GetColor( tList->GetMainColor() );
  Float_t rgba[4] = { 1, 0, 0, 1 };
  if (c) {
    rgba[0] = c->GetRed();
    rgba[1] = c->GetGreen();
    rgba[2] = c->GetBlue();
  }


  // Loop over L1 triggers and see which ones fired, and plot them
  std::vector<l1extra::L1ParticleMap>::const_iterator itrigger = particleMaps->begin(),
    itriggerEnd = particleMaps->end();
  for ( ; itrigger != itriggerEnd; ++itrigger ) {
     
  
    // Get a reference to the L1ParticleMap in question
    l1extra::L1ParticleMap const & particleMap = *itrigger; 
    if ( ! particleMap.triggerDecision() ) {
      std::cout << "The L1 trigger " << itrigger->triggerName() << " did not accept this event!" << std::endl;
    } 
    // Now do work if the event was accepted
    else {

      // Make a list of triggered object 4-vectors
      std::vector<reco::Particle::LorentzVector> p4s;

      // Get each list of triggered objects
      l1extra::L1EmParticleVectorRef     const & emTrigs = particleMap.emParticles();
      l1extra::L1MuonParticleVectorRef   const & muonTrigs = particleMap.muonParticles();
      l1extra::L1JetParticleVectorRef    const & jetTrigs = particleMap.jetParticles();
      l1extra::L1EtMissParticleRefProd   const & etMissTrigs = particleMap.etMissParticle();
      
      // Ready to loop over the triggered objects
      l1extra::L1EmParticleVectorRef::const_iterator emIt = emTrigs.begin(),
	emEnd = emTrigs.end();
      l1extra::L1MuonParticleVectorRef::const_iterator muonIt = muonTrigs.begin(),
	muonEnd = muonTrigs.end();
      l1extra::L1JetParticleVectorRef::const_iterator jetIt = jetTrigs.begin(),
	jetEnd = jetTrigs.end();

      // Loop over triggered objects and make some 4-vectors
      for ( ; emIt != emEnd; ++emIt ) {
	p4s.push_back( (*emIt)->p4() );
      }
      for ( ; muonIt != muonEnd; ++muonIt ) {
	p4s.push_back( (*muonIt)->p4() );
      }
      for ( ; jetIt != jetEnd; ++jetIt ) {
	p4s.push_back( (*jetIt)->p4() );
      }
      if ( etMissTrigs.isNonnull() )
	p4s.push_back ( etMissTrigs->p4() );

      // Loop over the triggered objects and fill histogram
      std::vector<reco::Particle::LorentzVector>::const_iterator p4It = p4s.begin(),
	p4End = p4s.end();
      for ( ; p4It != p4End; ++p4It, ++counter ) {

	// Get eve container
	TEveElementList* container = new TEveElementList( counter.str().c_str() );
	TGeoTube *shape = new TGeoTube(0.48, 0.5, 0.0001);
	TEveTrans t;
	t.RotateLF(1,2,M_PI/2);

	// Find eta and phi based on object type
	double eta = p4It->eta();
	double phi = p4It->phi();

	
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
      }

    }// End if we fired the trigger

  }// End loop over triggers
  

}


