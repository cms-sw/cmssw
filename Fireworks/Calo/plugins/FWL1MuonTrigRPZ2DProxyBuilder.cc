// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1MuonTrigRPZ2DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWL1MuonTrigRPZ2DProxyBuilder.cc,v 1.1 2009/01/19 14:40:05 amraktad Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveManager.h"
#include "TEveScalableStraightLineSet.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "Fireworks/Core/interface/FWDisplayEvent.h"

class FWL1MuonTrigRPZ2DProxyBuilder : public FWRPZ2DDataProxyBuilder
{

public:
   FWL1MuonTrigRPZ2DProxyBuilder();
   virtual ~FWL1MuonTrigRPZ2DProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void buildRhoPhi(const FWEventItem* iItem,
                            TEveElementList** product);

   virtual void buildRhoZ(const FWEventItem* iItem,
                          TEveElementList** product);

   double getTheta( double eta ) {
      return 2*atan(exp(-eta));
   }

   FWL1MuonTrigRPZ2DProxyBuilder(const FWL1MuonTrigRPZ2DProxyBuilder&);    // stop default

   const FWL1MuonTrigRPZ2DProxyBuilder& operator=(const FWL1MuonTrigRPZ2DProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWL1MuonTrigRPZ2DProxyBuilder::FWL1MuonTrigRPZ2DProxyBuilder()
{
}

// FWL1MuonTrigRPZ2DProxyBuilder::FWL1MuonTrigRPZ2DProxyBuilder(const FWL1MuonTrigRPZ2DProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWL1MuonTrigRPZ2DProxyBuilder::~FWL1MuonTrigRPZ2DProxyBuilder()
{
}

//
// mmuonber functions
//
void
FWL1MuonTrigRPZ2DProxyBuilder::buildRhoPhi(const FWEventItem* iItem,
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

   // Get the particle map collection for L1MuonParticles
   l1extra::L1MuonParticleCollection const * triggerColl=0;
   iItem->get(triggerColl);
   if(0==triggerColl) return;

   double r_ecal = 126;
   fw::NamedCounter counter("l1muontrigs");

   // Ready to loop over the triggered objects
   l1extra::L1MuonParticleCollection::const_iterator trigIt = triggerColl->begin(),
                                                     trigEnd = triggerColl->end();
   // Loop over triggered objects and make some 4-vectors
   for ( ; trigIt != trigEnd; ++trigIt ) {
      const unsigned int nBuffer = 1024;
      char title[nBuffer];
      snprintf(title, nBuffer, "L1 Muon %d, Et: %0.1f GeV",counter.index(),trigIt->et());
      TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
      container->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

      double phi = trigIt->phi();
      double size = trigIt->pt();

      TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
      marker->SetLineWidth(1);
      marker->SetLineStyle(7);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
      marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
      container->AddElement(marker);
      container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );

      tList->AddElement(container);

   } // end loop over muon particle objects

}


void
FWL1MuonTrigRPZ2DProxyBuilder::buildRhoZ(const FWEventItem* iItem,
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

   // Get the particle map collection for L1MuonParticles
   l1extra::L1MuonParticleCollection const * triggerColl=0;
   iItem->get(triggerColl);
   if(0==triggerColl) return;

   double z_ecal = 306; // ECAL endcap inner surface
   double r_ecal = 126;
   double transition_angle = atan(r_ecal/z_ecal);
   fw::NamedCounter counter("l1muontrigs");


   // Ready to loop over the triggered objects
   l1extra::L1MuonParticleCollection::const_iterator trigIt = triggerColl->begin(),
                                                     trigEnd = triggerColl->end();
   // Loop over triggered objects and make some 4-vectors
   for ( ; trigIt != trigEnd; ++trigIt ) {
      const unsigned int nBuffer = 1024;
      char title[nBuffer];
      snprintf(title, nBuffer, "L1 Muon %d, Et: %0.1f GeV",counter.index(),trigIt->et());
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
      marker->SetLineWidth(1);
      marker->SetLineStyle(7);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      marker->SetScaleCenter(0., (trigIt->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta));
      marker->AddLine(0., (trigIt->phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
                      0., (trigIt->phi()>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
      container->AddElement( marker );
      container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
      tList->AddElement(container);

   } // end loop over muon particle objects

}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWL1MuonTrigRPZ2DProxyBuilder,l1extra::L1MuonParticleCollection,"L1-Muons");
