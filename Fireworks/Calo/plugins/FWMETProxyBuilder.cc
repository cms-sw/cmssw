// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWMETProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWMETProxyBuilder.cc,v 1.4 2010/04/16 10:59:50 amraktad Exp $
//

// system include files
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"

#include "DataFormats/METReco/interface/MET.h"

class FWMETProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWMETProxyBuilder() {}
   virtual ~FWMETProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMETProxyBuilder(const FWMETProxyBuilder&);    // stop default
   const FWMETProxyBuilder& operator=(const FWMETProxyBuilder&);    // stop default

   virtual void build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWMETProxyBuilder::build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   double r_ecal = 126;
   double phi = iData.phi();
   double size = iData.et()*2;

   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
   marker->SetLineWidth(2);
   marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
   const double dx = 0.9*size*0.1;
   const double dy = 0.9*size*cos(0.1);
   marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0,
		    (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
   marker->AddLine( dx*sin(phi) + (dy+r_ecal)*cos(phi), -dx*cos(phi) + (dy+r_ecal)*sin(phi), 0,
		    (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
   marker->AddLine( -dx*sin(phi) + (dy+r_ecal)*cos(phi), dx*cos(phi) + (dy+r_ecal)*sin(phi), 0,
		    (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);

   oItemHolder.AddElement(marker);
}

class FWMETGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWMETGlimpseProxyBuilder() {}
   virtual ~FWMETGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMETGlimpseProxyBuilder(const FWMETGlimpseProxyBuilder&);    // stop default
   const FWMETGlimpseProxyBuilder& operator=(const FWMETGlimpseProxyBuilder&);    // stop default

   virtual void build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWMETGlimpseProxyBuilder::build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   double phi = iData.phi();
   double size = iData.et();
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
   marker->SetLineWidth(2);
   marker->AddLine( 0, 0, 0, size*cos(phi), size*sin(phi), 0);
   marker->AddLine( size*0.9*cos(phi+0.03), size*0.9*sin(phi+0.03), 0, size*cos(phi), size*sin(phi), 0);
   marker->AddLine( size*0.9*cos(phi-0.03), size*0.9*sin(phi-0.03), 0, size*cos(phi), size*sin(phi), 0);
   oItemHolder.AddElement(marker);
}

class FWMETLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWMETLegoProxyBuilder() {}
   virtual ~FWMETLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMETLegoProxyBuilder(const FWMETLegoProxyBuilder&);    // stop default
   const FWMETLegoProxyBuilder& operator=(const FWMETLegoProxyBuilder&);    // stop default

   virtual void build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWMETLegoProxyBuilder::build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   TEveStraightLineSet* mainLine = new TEveStraightLineSet( "MET phi" );
   mainLine->AddLine(-5.191, iData.phi(), 0.1, 5.191, iData.phi(), 0.1 );
   oItemHolder.AddElement( mainLine );

   double phi = iData.phi();
   phi = phi > 0 ? phi - M_PI : phi + M_PI;
   TEveStraightLineSet* secondLine = new TEveStraightLineSet( "MET opposite phi" );
   secondLine->SetLineStyle(7);
   secondLine->AddLine(-5.191, phi, 0.1, 5.191, phi, 0.1 );
   oItemHolder.AddElement( secondLine );
}

REGISTER_FWPROXYBUILDER(FWMETProxyBuilder, reco::MET, "MET", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWMETGlimpseProxyBuilder, reco::MET, "MET", FWViewType::kGlimpseBit);
REGISTER_FWPROXYBUILDER(FWMETLegoProxyBuilder, reco::MET, "MET", FWViewType::kLegoBit);
