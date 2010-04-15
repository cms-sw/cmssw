// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1JetParticleProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWL1JetParticleProxyBuilder.cc,v 1.1 2010/04/15 10:04:39 yana Exp $
//

// system include files
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

class FWL1JetParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1JetParticle>
{
public:
   FWL1JetParticleProxyBuilder() {}
   virtual ~FWL1JetParticleProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1JetParticleProxyBuilder(const FWL1JetParticleProxyBuilder&);    // stop default
   const FWL1JetParticleProxyBuilder& operator=(const FWL1JetParticleProxyBuilder&);    // stop default
  
   virtual void build( const l1extra::L1JetParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWL1JetParticleProxyBuilder::build( const l1extra::L1JetParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
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
   
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("l1JetParticle");
   marker->SetLineWidth( 2 );
   marker->SetLineStyle( 2 );
   marker->AddLine( r*cos(phi)*sin(theta), r*sin(phi)*sin(theta), r*cos(theta),
		    (r+size)*cos(phi)*sin(theta), (r+size)*sin(phi)*sin(theta), (r+size)*cos(theta) );
   oItemHolder.AddElement( marker );
}

class FWL1JetParticleLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1JetParticle>
{
public:
   FWL1JetParticleLegoProxyBuilder() {}
   virtual ~FWL1JetParticleLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1JetParticleLegoProxyBuilder(const FWL1JetParticleLegoProxyBuilder&);    // stop default
   const FWL1JetParticleLegoProxyBuilder& operator=(const FWL1JetParticleLegoProxyBuilder&);    // stop default
  
   virtual void build( const l1extra::L1JetParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWL1JetParticleLegoProxyBuilder::build( const l1extra::L1JetParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   const unsigned int nLineSegments = 6;
   const double jetRadius = 0.5;

   char title[1024];
   sprintf(title,"L1 Jet %d, Et: %0.1f GeV", iIndex,iData.et());
   TEveStraightLineSet* container = new TEveStraightLineSet( "l1JetParticle", title );
   for( unsigned int iphi = 0; iphi < nLineSegments; ++iphi ) {
      container->AddLine(iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*iphi),
			 iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*iphi),
			 0.1,
			 iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*(iphi+1)),
			 iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*(iphi+1)),
			 0.1);
   }
   oItemHolder.AddElement( container );
}


REGISTER_FWPROXYBUILDER(FWL1JetParticleProxyBuilder, l1extra::L1JetParticle, "L1JetParticle", FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWL1JetParticleLegoProxyBuilder, l1extra::L1JetParticle, "L1JetParticle", FWViewType::kLegoBit);
