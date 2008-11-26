// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1JetTrigProxyRhoPhiZ2DBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: L1JetTrigProxyRhoPhiZ2DBuilder.cc,v 1.10 2008/11/06 22:05:21 amraktad Exp $
//
//
// system include files
#include "TEveGeoNode.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEvePointSet.h"
#include "TEveScalableStraightLineSet.h"
#include "TEveCompound.h"

#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>


// user include files
#include "Fireworks/Calo/interface/L1JetTrigProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
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
L1JetTrigProxyRhoPhiZ2DBuilder::L1JetTrigProxyRhoPhiZ2DBuilder()
{
}

// L1JetTrigProxyRhoPhiZ2DBuilder::L1JetTrigProxyRhoPhiZ2DBuilder(const L1JetTrigProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

L1JetTrigProxyRhoPhiZ2DBuilder::~L1JetTrigProxyRhoPhiZ2DBuilder()
{
}

//
// mjetber functions
//
void
L1JetTrigProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{


  TEveElementList* tList = *product;

  // Make the eve element list
  if(0 == tList) {
    tList =  new TEveElementList(iItem->name().c_str(),"L1 RhoPhi",true);
    *product = tList;
    tList->SetMainColor(iItem->defaultDisplayProperties().color());
    gEve->AddElement(tList);
  } else {
    tList->DestroyElements();
  }



  // Get the particle map collection for L1JetParticles
  l1extra::L1JetParticleCollection const * triggerColl=0;
  iItem->get(triggerColl);
  if(0==triggerColl) return;

  // make a counter
   double r_ecal = 126;
   fw::NamedCounter counter("l1jettrigs");

   // Ready to loop over the triggered objects
   l1extra::L1JetParticleCollection::const_iterator trigIt = triggerColl->begin(),
     trigEnd = triggerColl->end();
   // Loop over triggered objects and make some 4-vectors
   for ( ; trigIt != trigEnd; ++trigIt, ++counter ) {
      const unsigned int nBuffer = 1024;
      char title[nBuffer];
      snprintf(title, nBuffer,"L1 Jet %d, Et: %0.1f GeV",counter.index(),trigIt->et());
     TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
     container->OpenCompound();
     //guarantees that CloseCompound will be called no matter what happens
     boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

     double phi = trigIt->phi();
     double size = trigIt->pt();

      TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
      marker->SetLineWidth(2);
      marker->SetLineStyle(2);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
      marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
      container->AddElement(marker);
      container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );

     tList->AddElement(container);

   }// end loop over jet particle objects

}


void
L1JetTrigProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					    TEveElementList** product)
{


  TEveElementList* tList = *product;

  // Make the eve element list
  if(0 == tList) {
    tList =  new TEveElementList(iItem->name().c_str(),"L1 RhoZ",true);
    *product = tList;
    tList->SetMainColor(iItem->defaultDisplayProperties().color());
    gEve->AddElement(tList);
  } else {
    tList->DestroyElements();
  }



  // Get the particle map collection for L1JetParticles
  l1extra::L1JetParticleCollection const * triggerColl=0;
  iItem->get(triggerColl);
  if(0==triggerColl) return;

   double z_ecal = 306; // ECAL endcap inner surface
   double r_ecal = 126;
   double transition_angle = atan(r_ecal/z_ecal);
   fw::NamedCounter counter("l1jettrigs");


   // Ready to loop over the triggered objects
   l1extra::L1JetParticleCollection::const_iterator trigIt = triggerColl->begin(),
     trigEnd = triggerColl->end();
   // Loop over triggered objects and make some 4-vectors
   for ( ; trigIt != trigEnd; ++trigIt, ++counter ) {
      const unsigned int nBuffer = 1024;
      char title[nBuffer];
      snprintf(title, nBuffer,"L1 Jet %d, Et: %0.1f GeV",counter.index(),trigIt->et());
     TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
     container->OpenCompound();
     //guarantees that CloseCompound will be called no matter what happens
     boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

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

      TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
      marker->SetLineWidth(2);
      marker->SetLineStyle(2);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      marker->SetScaleCenter(0., (trigIt->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta));
      marker->AddLine(0., (trigIt->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
		      0., (trigIt->phi()>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
      container->AddElement( marker );
      container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
     tList->AddElement(container);



   }// end loop over jet particle objects

}

REGISTER_FWRPZDATAPROXYBUILDERBASE(L1JetTrigProxyRhoPhiZ2DBuilder,l1extra::L1JetParticleCollection,"L1-Jets");
