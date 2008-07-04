// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1EmTrigProxyRhoPhiZ2DBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: L1EmTrigProxyRhoPhiZ2DBuilder.cc,v 1.3 2008/06/28 22:17:33 dmytro Exp $
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
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Calo/interface/L1EmTrigProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
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
L1EmTrigProxyRhoPhiZ2DBuilder::L1EmTrigProxyRhoPhiZ2DBuilder()
{
}

// L1EmTrigProxyRhoPhiZ2DBuilder::L1EmTrigProxyRhoPhiZ2DBuilder(const L1EmTrigProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

L1EmTrigProxyRhoPhiZ2DBuilder::~L1EmTrigProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void 
L1EmTrigProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
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
   


  // Get the particle map collection for L1EmParticles
  l1extra::L1EmParticleCollection const * triggerColl=0;
  iItem->get(triggerColl);
  if(0==triggerColl) {
    std::cout <<"Failed to get L1EmTrig particle collection"<<std::endl;
    return;
  }


  // make a counter
   double r_ecal = 126;
   //double minJetEt = 15;
   double minJetEt = 0;
   fw::NamedCounter counter("l1emtrigs");

   // Ready to loop over the triggered objects
   l1extra::L1EmParticleCollection::const_iterator trigIt = triggerColl->begin(),
     trigEnd = triggerColl->end();
   // Loop over triggered objects and make some 4-vectors
   for ( ; trigIt != trigEnd; ++trigIt ) {


     TEveElementList* container = new TEveElementList( counter.str().c_str() );
     std::vector<double> p4phis; p4phis.push_back( trigIt->phi() );
     std::pair<double,double> phiRange = fw::getPhiRange( p4phis, trigIt->phi() );
     double min_phi = phiRange.first-M_PI/36/2;
     double max_phi = phiRange.second+M_PI/36/2;
      
     double phi = trigIt->phi();

     double size = trigIt->pt();
     TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
     TEveGeoShapeExtract *sc = fw::getShapeExtract( "spread", sc_box, iItem->defaultDisplayProperties().color() );
      
     if ( trigIt->pt() > minJetEt ) {
       TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
       marker->SetLineWidth(4);
       marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
       TEveElement* element = TEveGeoShape::ImportShapeExtract(sc, 0);
       element->SetPickable(kTRUE);
       container->AddElement(element);
       marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
       marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
       container->AddElement(marker);
     }
     tList->AddElement(container);

   }// end loop over em particle objects

}


void 
L1EmTrigProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
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
 


  // Get the particle map collection for L1EmParticles
  l1extra::L1EmParticleCollection const * triggerColl=0;
  iItem->get(triggerColl);
  if(0==triggerColl) {
    std::cout <<"Failed to get L1EmTrig particle collection"<<std::endl;
    return;
  }


   // NOTE:
   //      We derive eta bin size from xbins array used for LEGO assuming that all 82
   //      eta bins are accounted there. 
   assert ( sizeof(fw3dlego::xbins)/sizeof(*fw3dlego::xbins) == 82+1 );
   static const std::vector<std::pair<double,double> > thetaBins = ECalCaloTowerProxyRhoPhiZ2DBuilder::getThetaBins();


   double z_ecal = 306; // ECAL endcap inner surface
   double r_ecal = 126;
   double transition_angle = atan(r_ecal/z_ecal);
   //double minJetEt = 15;
   double minJetEt = 0;
   fw::NamedCounter counter("l1emtrigs");


   // Ready to loop over the triggered objects
   l1extra::L1EmParticleCollection::const_iterator trigIt = triggerColl->begin(),
     trigEnd = triggerColl->end();
   // Loop over triggered objects and make some 4-vectors
   for ( ; trigIt != trigEnd; ++trigIt ) {


     TEveElementList* container = new TEveElementList( counter.str().c_str() );

      
     // 	 double max_theta = thetaBins[iEtaRange.first].first;
     // 	 double min_theta = thetaBins[iEtaRange.second].second;;

     double max_theta = trigIt->theta() + 0.0001;
     double min_theta = trigIt->theta() - 0.0001;
      
     double theta = trigIt->theta();
      
     // distance from the origin of the jet centroid
     // energy is measured from this point
     // if jet is made of a single tower, the length of the jet will 
     // be identical to legth of the displayed tower
     double r(0); 
     if ( theta < transition_angle || M_PI-theta < transition_angle )
       r = z_ecal/fabs(cos(theta));
     else
       r = r_ecal/sin(theta);
      
     double size = trigIt->pt();
      
     if ( trigIt->pt() > minJetEt ) {
       TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
       marker->SetLineWidth(4);
       marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
       marker->SetScaleCenter( 0., (trigIt->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );
       marker->AddLine(0., (trigIt->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
		       0., (trigIt->phi()>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
       container->AddElement( marker );
       fw::addRhoZEnergyProjection( container, r_ecal, z_ecal, min_theta-0.003, max_theta+0.003, 
				    trigIt->phi(), iItem->defaultDisplayProperties().color() );
     }
     tList->AddElement(container);



   }// end loop over em particle objects

}

REGISTER_FWRPZ2DDATAPROXYBUILDER(L1EmTrigProxyRhoPhiZ2DBuilder,l1extra::L1EmParticleCollection,"L1EmTrig");
