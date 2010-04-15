// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1EmParticleProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWL1EmParticleProxyBuilder.cc,v 1.1 2010/04/14 15:52:18 yana Exp $
//

// system include files
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"

class FWL1EmParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EmParticle>
{
public:
   FWL1EmParticleProxyBuilder() {}
   virtual ~FWL1EmParticleProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1EmParticleProxyBuilder(const FWL1EmParticleProxyBuilder&);    // stop default
   const FWL1EmParticleProxyBuilder& operator=(const FWL1EmParticleProxyBuilder&);    // stop default
  
   virtual void build( const l1extra::L1EmParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWL1EmParticleProxyBuilder::build( const l1extra::L1EmParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   double scale = 10;
   double r_ecal = 126;
   double z_ecal = 306; // ECAL endcap inner surface
   double transition_angle = atan(r_ecal/z_ecal);
   double phi = iData.phi();
   double theta = iData.theta();
   double size = iData.pt() * scale;

   // distance from the origin of the jet centroid
   // energy is measured from this point
   // if jet is made of a single tower, the length of the jet will
   // be identical to legth of the displayed tower
   double r(0);
   if( theta < transition_angle || M_PI-theta < transition_angle )
     r = z_ecal/fabs(cos(theta));
   else
     r = r_ecal/sin(theta);
   
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("l1EmParticle");
   marker->SetLineWidth( 2 );
   marker->SetLineStyle( 2 );
   marker->AddLine( r*cos(phi)*sin(theta), r*sin(phi)*sin(theta), r*cos(theta),
		    (r+size)*cos(phi)*sin(theta), (r+size)*sin(phi)*sin(theta), (r+size)*cos(theta) );
   oItemHolder.AddElement( marker );
}

REGISTER_FWPROXYBUILDER(FWL1EmParticleProxyBuilder, l1extra::L1EmParticle, "L1EmParticle", FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
