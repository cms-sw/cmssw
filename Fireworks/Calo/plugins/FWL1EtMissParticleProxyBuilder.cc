// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1EtMissParticleProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWL1EtMissParticleProxyBuilder.cc,v 1.3 2010/04/20 20:49:40 amraktad Exp $
//

// system include files
#include "TEveCompound.h"
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

class FWL1EtMissParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EtMissParticle>
{
public:
   FWL1EtMissParticleProxyBuilder() {}
   virtual ~FWL1EtMissParticleProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1EtMissParticleProxyBuilder(const FWL1EtMissParticleProxyBuilder&);    // stop default
   const FWL1EtMissParticleProxyBuilder& operator=(const FWL1EtMissParticleProxyBuilder&);    // stop default
  
   virtual void build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder );
};

void
FWL1EtMissParticleProxyBuilder::build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) 
{
   double scale = 10;
   float r_ecal = fireworks::Context::s_ecalR;
   float z_ecal = fireworks::Context::s_ecalZ;
   float transition_angle = fireworks::Context::s_transitionAngle;
   double phi = iData.phi();
   double theta = iData.theta();
   double size = iData.pt() * scale;

   // distance from the origin of the jet centroid
   // energy is measured from this point
   // if jet is made of a single tower, the length of the jet will
   // be identical to legth of the displayed tower
   double r(0);
   if( theta < transition_angle || M_PI-theta < transition_angle )
     r = z_ecal/fabs( cos( theta ));
   else
     r = r_ecal/sin( theta );
   
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("l1EtMissParticle");
   marker->SetLineWidth( 2 );
   marker->SetLineStyle( 2 );
   marker->AddLine( r*cos(phi)*sin(theta), r*sin(phi)*sin(theta), r*cos(theta),
		    (r+size)*cos(phi)*sin(theta), (r+size)*sin(phi)*sin(theta), (r+size)*cos(theta) );
   setupAddElement(marker, &oItemHolder);
}

class FWL1EtMissParticleGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EtMissParticle>
{
public:
   FWL1EtMissParticleGlimpseProxyBuilder() {}
   virtual ~FWL1EtMissParticleGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1EtMissParticleGlimpseProxyBuilder(const FWL1EtMissParticleGlimpseProxyBuilder&);    // stop default
   const FWL1EtMissParticleGlimpseProxyBuilder& operator=(const FWL1EtMissParticleGlimpseProxyBuilder&);    // stop default
  
   virtual void build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder );
};

void
FWL1EtMissParticleGlimpseProxyBuilder::build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) 
{
   char title[1024];
   sprintf( title, "L1 MET: %0.1f GeV", iData.et() );
   TEveCompound* container = new TEveCompound( "L1EtMissParticle", title );
   container->OpenCompound();
   //guarantees that CloseCompound will be called no matter what happens
   boost::shared_ptr<TEveCompound> sentry( container, boost::mem_fn( &TEveCompound::CloseCompound ));
   
   double phi = iData.phi();
   double size = iData.et();
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet( "L1EtMissParticle" );
   marker->SetLineWidth( 1 );
   marker->SetLineStyle( 2 );
   marker->AddLine( 0, 0, 0, size*cos(phi), size*sin(phi), 0);
   marker->AddLine( size*0.9*cos(phi+0.03), size*0.9*sin(phi+0.03), 0, size*cos(phi), size*sin(phi), 0);
   marker->AddLine( size*0.9*cos(phi-0.03), size*0.9*sin(phi-0.03), 0, size*cos(phi), size*sin(phi), 0);
   setupAddElement(marker, container);

   setupAddElement(container, &oItemHolder);
}

class FWL1EtMissParticleLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1EtMissParticle>
{
public:
   FWL1EtMissParticleLegoProxyBuilder() {}
   virtual ~FWL1EtMissParticleLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWL1EtMissParticleLegoProxyBuilder(const FWL1EtMissParticleLegoProxyBuilder&);    // stop default
   const FWL1EtMissParticleLegoProxyBuilder& operator=(const FWL1EtMissParticleLegoProxyBuilder&);    // stop default
  
   virtual void build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder );
};

void
FWL1EtMissParticleLegoProxyBuilder::build( const l1extra::L1EtMissParticle& iData, unsigned int iIndex, TEveElement& oItemHolder ) 
{
   char title[1024];
   sprintf(title, "L1 MET: %0.1f GeV", iData.et());
   TEveCompound* container = new TEveCompound( "L1EtMissParticle", title );
   container->OpenCompound();
   //guarantees that CloseCompound will be called no matter what happens
   boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));
   
   TEveStraightLineSet* mainLine = new TEveStraightLineSet( "MET phi" );
   mainLine->AddLine(-5.191, iData.phi(), 0.1, 5.191, iData.phi(), 0.1 );
   setupAddElement(mainLine, container);
   
   double phi = iData.phi();
   phi = phi > 0 ? phi - M_PI : phi + M_PI;
   TEveStraightLineSet* secondLine = new TEveStraightLineSet( "MET opposite phi" );
   secondLine->SetLineStyle(7);
   secondLine->AddLine(-5.191, phi, 0.1, 5.191, phi, 0.1 );
   setupAddElement(secondLine, container);

   setupAddElement(container, &oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWL1EtMissParticleProxyBuilder, l1extra::L1EtMissParticle, "L1EtMissParticle", FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWL1EtMissParticleGlimpseProxyBuilder, l1extra::L1EtMissParticle, "L1EtMissParticle", FWViewType::kGlimpseBit);
REGISTER_FWPROXYBUILDER(FWL1EtMissParticleLegoProxyBuilder, l1extra::L1EtMissParticle, "L1EtMissParticle", FWViewType::kLegoBit);
