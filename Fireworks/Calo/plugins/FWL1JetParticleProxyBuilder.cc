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
// $Id: FWL1JetParticleProxyBuilder.cc,v 1.6 2010/08/30 15:42:32 amraktad Exp $
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
  
   virtual void build( const l1extra::L1JetParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*);
};

void
FWL1JetParticleProxyBuilder::build( const l1extra::L1JetParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*) 
{
   double scale = 10;
   double phi = iData.phi();
   double theta = iData.theta();
   double size = iData.pt() * scale;

   // distance from the origin of the jet centroid
   // energy is measured from this point
   // if jet is made of a single tower, the length of the jet will
   // be identical to legth of the displayed tower
   double r(0);
   if( theta < context().caloTransAngle() || M_PI-theta < context().caloTransAngle())
      r = context().caloZ2()/fabs(cos(theta));
   else
      r = context().caloR1()/sin(theta);
   
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("l1JetParticle");
   marker->SetLineWidth( 2 );
   marker->SetLineStyle( 2 );
   marker->AddLine( r*cos(phi)*sin(theta), r*sin(phi)*sin(theta), r*cos(theta),
		    (r+size)*cos(phi)*sin(theta), (r+size)*sin(phi)*sin(theta), (r+size)*cos(theta) );
   setupAddElement(marker, &oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWL1JetParticleProxyBuilder, l1extra::L1JetParticle, "L1JetParticle", FWViewType::kAllRPZBits);

//==============================================================================

class FWL1JetParticleLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1JetParticle>
{
public:
   FWL1JetParticleLegoProxyBuilder() {}
   virtual ~FWL1JetParticleLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1JetParticleLegoProxyBuilder(const FWL1JetParticleLegoProxyBuilder&);    // stop default
   const FWL1JetParticleLegoProxyBuilder& operator=(const FWL1JetParticleLegoProxyBuilder&);    // stop default
  
   virtual void build( const l1extra::L1JetParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*);
};

void
FWL1JetParticleLegoProxyBuilder::build( const l1extra::L1JetParticle& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*) 
{
   const unsigned int nLineSegments = 6;
   const double jetRadius = 0.5;

   TEveStraightLineSet* container = new TEveStraightLineSet( "l1JetParticle");
   for( unsigned int iphi = 0; iphi < nLineSegments; ++iphi ) {
      container->AddLine(iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*iphi),
			 iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*iphi),
			 0.1,
			 iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*(iphi+1)),
			 iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*(iphi+1)),
			 0.1);
   }
   setupAddElement(container, &oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWL1JetParticleLegoProxyBuilder, l1extra::L1JetParticle, "L1JetParticle", FWViewType::kLegoBit);
