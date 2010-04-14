// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1EmTrigRPZ2DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWL1EmTrigRPZ2DProxyBuilder.cc,v 1.3 2009/10/27 01:43:28 dmytro Exp $
//

// system include files
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/Candidate/interface/LeafCandidate.h"

class FWL1EmTrigRPZ2DProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::LeafCandidate>
{
public:
   FWL1EmTrigRPZ2DProxyBuilder() {}
   virtual ~FWL1EmTrigRPZ2DProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   void build(const reco::LeafCandidate& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   FWL1EmTrigRPZ2DProxyBuilder(const FWL1EmTrigRPZ2DProxyBuilder&);    // stop default

   const FWL1EmTrigRPZ2DProxyBuilder& operator=(const FWL1EmTrigRPZ2DProxyBuilder&);    // stop default
};

void
FWL1EmTrigRPZ2DProxyBuilder::build(const reco::LeafCandidate& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   double z_ecal = 306; // ECAL endcap inner surface
   double r_ecal = 126;
   double transition_angle = atan(r_ecal/z_ecal);
//    fw::NamedCounter counter("l1emtrigs");

//    const unsigned int nBuffer = 1024;
//    char title[nBuffer];
// //    snprintf(title, nBuffer, "L1 em trig %d, Et: %0.1f GeV",counter.index(),iData->et());
//    TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
//    container->OpenCompound();
//    //guarantees that CloseCompound will be called no matter what happens
//    boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

   double theta = iData.theta();

   // distance from the origin of the jet centroid
   // energy is measured from this point
   // if jet is made of a single tower, the length of the jet will
   // be identical to legth of the displayed tower
   double r(0);
   if ( theta < transition_angle || M_PI-theta < transition_angle )
     r = z_ecal/fabs(cos(theta));
   else
     r = r_ecal/sin(theta);

   double size = iData.pt();

   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
   marker->SetLineWidth(2);
   marker->SetLineStyle(2);
   marker->SetLineColor( item()->defaultDisplayProperties().color() );
   marker->SetScaleCenter( 0., (iData.phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );
   marker->AddLine(0., (iData.phi()>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
		   0., (iData.phi()>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
   oItemHolder.AddElement( marker );
}

REGISTER_FWPROXYBUILDER(FWL1EmTrigRPZ2DProxyBuilder, reco::LeafCandidate, "LeafCandidate", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
