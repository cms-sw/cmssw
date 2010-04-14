// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1EmTrigProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWL1EmTrigProxyBuilder.cc,v 1.3 2009/10/27 01:43:28 dmytro Exp $
//

// system include files
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"

class FWL1EmTrigProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EmParticle>
{
public:
   FWL1EmTrigProxyBuilder() {}
   virtual ~FWL1EmTrigProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1EmTrigProxyBuilder(const FWL1EmTrigProxyBuilder&);    // stop default
   const FWL1EmTrigProxyBuilder& operator=(const FWL1EmTrigProxyBuilder&);    // stop default
  
   virtual void build( const l1extra::L1EmParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

// void
// FWL1EmTrigProxyBuilder::buildRhoPhi(const FWEventItem* iItem,
//                                          TEveElementList** product)
// {
//    TEveElementList* tList = *product;

//    // Make the eve element list
//    if(0 == tList) {
//       tList =  new TEveElementList(iItem->name().c_str(),"L1 RhoPhi",true);
//       *product = tList;
//       tList->SetMainColor(iItem->defaultDisplayProperties().color());
//    } else {
//       tList->DestroyElements();
//    }



//    // Get the particle map collection for L1EmParticles
//    l1extra::L1EmParticleCollection const * triggerColl=0;
//    iItem->get(triggerColl);
//    if(0==triggerColl) return;

//    // make a counter
//    double r_ecal = 126;
//    fw::NamedCounter counter("l1emtrigs");

//    // Ready to loop over the triggered objects
//    l1extra::L1EmParticleCollection::const_iterator trigIt = triggerColl->begin(),
//                                                    trigEnd = triggerColl->end();
//    // Loop over triggered objects and make some 4-vectors
//    for ( ; trigIt != trigEnd; ++trigIt ) {
//       const unsigned int nBuffer = 1024;
//       char title[nBuffer];
//       snprintf(title, nBuffer, "L1 em trig %d, Et: %0.1f GeV",counter.index(),trigIt->et());
//       TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
//       container->OpenCompound();
//       //guarantees that CloseCompound will be called no matter what happens
//       boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

void
FWL1EmTrigProxyBuilder::build( const l1extra::L1EmParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   double z_ecal = 306; // ECAL endcap inner surface
   double phi = iData.phi();
   double size = iData.pt();

   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
   marker->SetLineWidth(2);
   marker->SetLineStyle(2);
   marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
   marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
   oItemHolder.AddElement(marker);
}


// void
// FWL1EmTrigProxyBuilder::build( const l1extra::L1EmParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
// {
//    double z_ecal = 306; // ECAL endcap inner surface
//    double r_ecal = 126;
//    double transition_angle = atan(r_ecal/z_ecal);
// //    fw::NamedCounter counter("l1emtrigs");

// //    const unsigned int nBuffer = 1024;
// //    char title[nBuffer];
// // //    snprintf(title, nBuffer, "L1 em trig %d, Et: %0.1f GeV",counter.index(),iData->et());
// //    TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
// //    container->OpenCompound();
// //    //guarantees that CloseCompound will be called no matter what happens
// //    boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

//    double theta = iData.theta();

//    // distance from the origin of the jet centroid
//    // energy is measured from this point
//    // if jet is made of a single tower, the length of the jet will
//    // be identical to legth of the displayed tower
//    double r(0);
//    if ( theta < transition_angle || M_PI-theta < transition_angle )
//      r = z_ecal/fabs(cos(theta));
//    else
//      r = r_ecal/sin(theta);

//    double size = iData.pt();

//    TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
//    marker->SetLineWidth(2);
//    marker->SetLineStyle(2);
//    marker->SetLineColor( item()->defaultDisplayProperties().color() );
//    marker->SetScaleCenter( 0., (iData.phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );
//    marker->AddLine(0., (iData.phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
// 		   0., (iData.phi()>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
//    oItemHolder.AddElement( marker );
// //    container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
// //    container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
// //    oItemHolder.AddElement(container);
// }

REGISTER_FWPROXYBUILDER(FWL1EmTrigProxyBuilder, l1extra::L1EmParticle, "L1EmTrig", FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
