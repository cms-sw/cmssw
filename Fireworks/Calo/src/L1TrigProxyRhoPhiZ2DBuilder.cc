// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1TrigProxyRhoPhiZ2DBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: L1TrigProxyRhoPhiZ2DBuilder.cc,v 1.8 2008/05/26 14:23:58 dmytro Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"

// user include files
#include "Fireworks/Calo/interface/L1TrigProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMapFwd.h"
#include "Fireworks/Core/interface/FWDisplayEvent.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TrigProxyRhoPhiZ2DBuilder::L1TrigProxyRhoPhiZ2DBuilder()
{
}

// L1TrigProxyRhoPhiZ2DBuilder::L1TrigProxyRhoPhiZ2DBuilder(const L1TrigProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

L1TrigProxyRhoPhiZ2DBuilder::~L1TrigProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void 
L1TrigProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{


  TEveElementList* tList = *product;

  // Make the eve element list
  if(0 == tList) {
    tList =  new TEveElementList(iItem->name().c_str(),"L1 RhoPhi",true);
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
   double r_ecal = 126;
   double scale = FWDisplayEvent::getCaloScale();
   if ( scale < 0 ) scale = 2;
   //double minJetEt = 15;
   double minJetEt = 0;
   fw::NamedCounter counter("jet");


  // Loop over L1 triggers and see which ones fired, and plot them
   std::vector<l1extra::L1ParticleMap>::const_iterator itrigger = particleMaps->begin(),
    itriggerEnd = particleMaps->end();
  for ( ; itrigger != itriggerEnd; ++itrigger ) {
     
  
    // Get a reference to the L1ParticleMap in question
    const l1extra::L1ParticleMap& particleMap = *itrigger; 
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
	p4s.push_back( etMissTrigs->p4() );
     

      // Loop over the triggered objects and fill histogram
      std::vector<reco::Particle::LorentzVector>::const_iterator p4It = p4s.begin(),
	p4End = p4s.end();
      for ( ; p4It != p4End; ++p4It, ++counter ) {
	TEveElementList* container = new TEveElementList( counter.str().c_str() );
	std::vector<double> p4phis; p4phis.push_back( p4It->phi() );
	std::pair<double,double> phiRange = fw::getPhiRange( p4phis, p4It->phi() );
	double min_phi = phiRange.first-M_PI/36/2;
	double max_phi = phiRange.second+M_PI/36/2;
      
	double phi = p4It->phi();

	double size = scale*p4It->pt();
	TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
	TEveGeoShapeExtract *sc = fw::getShapeExtract( "spread", sc_box, iItem->defaultDisplayProperties().color() );
      
	if ( p4It->pt() > minJetEt ) {
	  TEveStraightLineSet* marker = new TEveStraightLineSet("energy");
	  marker->SetLineWidth(4);
	  marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
	  TEveElement* element = TEveGeoShape::ImportShapeExtract(sc, 0);
	  element->SetPickable(kTRUE);
	  container->AddElement(element);
	  marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
	  container->AddElement(marker);
	}
	tList->AddElement(container);

      }

    }// End if we fired the trigger

  }// End loop over triggers

}


void 
L1TrigProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					    TEveElementList** product)
{


  TEveElementList* tList = *product;

  // Make the eve element list
  if(0 == tList) {
    tList =  new TEveElementList(iItem->name().c_str(),"L1 RhoZ",true);
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
   

   // NOTE:
   //      We derive eta bin size from xbins array used for LEGO assuming that all 82
   //      eta bins are accounted there. 
   assert ( sizeof(fw3dlego::xbins)/sizeof(*fw3dlego::xbins) == 82+1 );
   static const std::vector<std::pair<double,double> > thetaBins = ECalCaloTowerProxyRhoPhiZ2DBuilder::getThetaBins();


   double scale = FWDisplayEvent::getCaloScale();
   if ( scale < 0 ) scale = 2;
   double z_ecal = 306; // ECAL endcap inner surface
   double r_ecal = 126;
   double transition_angle = atan(r_ecal/z_ecal);
   //double minJetEt = 15;
   double minJetEt = 0;
   fw::NamedCounter counter("jet");



   // Loop over L1 triggers and see which ones fired, and plot them
   std::vector<l1extra::L1ParticleMap>::const_iterator itrigger = particleMaps->begin(),
     itriggerEnd = particleMaps->end();
   for ( ; itrigger != itriggerEnd; ++itrigger ) {
     
  
     // Get a reference to the L1ParticleMap in question
     const l1extra::L1ParticleMap& particleMap = *itrigger; 
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
	 TEveElementList* container = new TEveElementList( counter.str().c_str() );

      
// 	 double max_theta = thetaBins[iEtaRange.first].first;
// 	 double min_theta = thetaBins[iEtaRange.second].second;;

	 double max_theta = p4It->theta() + 0.0001;
	 double min_theta = p4It->theta() - 0.0001;
      
	 double theta = p4It->theta();
      
	 // distance from the origin of the jet centroid
	 // energy is measured from this point
	 // if jet is made of a single tower, the length of the jet will 
	 // be identical to legth of the displayed tower
	 double r(0); 
	 if ( theta < transition_angle || M_PI-theta < transition_angle )
	   r = z_ecal/fabs(cos(theta));
	 else
	   r = r_ecal/sin(theta);
      
	 double size = scale*p4It->pt();
      
	 if ( p4It->pt() > minJetEt ) {
	   TEveStraightLineSet* marker = new TEveStraightLineSet("energy");
	   marker->SetLineWidth(4);
	   marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
	   marker->AddLine(0., (p4It->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
			   0., (p4It->phi()>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
	   container->AddElement( marker );
	   fw::addRhoZEnergyProjection( container, r_ecal, z_ecal, min_theta-0.003, max_theta+0.003, 
					p4It->phi(), iItem->defaultDisplayProperties().color() );
	 }
	 tList->AddElement(container);



       }

     }// End if we fired the trigger

   }// End loop over triggers

}
