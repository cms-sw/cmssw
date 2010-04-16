// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWJetProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWJetProxyBuilder.cc,v 1.2 2010/04/15 13:19:32 yana Exp $
//
#include "TEveStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Calo/interface/FW3DEveJet.h"
#include "Fireworks/Calo/interface/FWGlimpseEveJet.h"
#include "DataFormats/JetReco/interface/Jet.h"

class FWJetProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetProxyBuilder() {}
   virtual ~FWJetProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWJetProxyBuilder(const FWJetProxyBuilder&); // stop default

   const FWJetProxyBuilder& operator=(const FWJetProxyBuilder&); // stop default

   virtual void build(const reco::Jet& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
};

void
FWJetProxyBuilder::build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   FW3DEveJet* cone = new FW3DEveJet( iData, "jet", "jet");
   cone->SetPickable(kTRUE);
   cone->SetMainTransparency(75);

   oItemHolder.AddElement( cone );
}

class FWJetGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetGlimpseProxyBuilder() {}
   virtual ~FWJetGlimpseProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWJetGlimpseProxyBuilder(const FWJetGlimpseProxyBuilder&); // stop default
   const FWJetGlimpseProxyBuilder& operator=(const FWJetGlimpseProxyBuilder&); // stop default

   virtual void build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWJetGlimpseProxyBuilder::build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   FWGlimpseEveJet* cone = new FWGlimpseEveJet(&iData, "jet", "jet");
   oItemHolder.AddElement( cone );
   
   cone->SetPickable( kTRUE );
   cone->SetMainTransparency( 50 );
   cone->SetRnrSelf( item()->defaultDisplayProperties().isVisible() );
   cone->SetRnrChildren( item()->defaultDisplayProperties().isVisible() );
   cone->SetDrawConeCap( kFALSE );
   cone->SetMainTransparency( 50 );
}

class FWJetLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetLegoProxyBuilder() {}
   virtual ~FWJetLegoProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWJetLegoProxyBuilder(const FWJetLegoProxyBuilder&); // stop default
   const FWJetLegoProxyBuilder& operator=(const FWJetLegoProxyBuilder&); // stop default

   virtual void build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWJetLegoProxyBuilder::build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   TEveStraightLineSet* container = new TEveStraightLineSet("circle");
   oItemHolder.AddElement(container);

   const unsigned int nLineSegments = 20;
   const double jetRadius = 0.5;
   for( unsigned int iphi = 0; iphi < nLineSegments; ++iphi ) {
      container->AddLine(iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*iphi),
                         iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*iphi),
                         0.1,
                         iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*(iphi+1)),
                         iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*(iphi+1)),
                         0.1);
   }
}

REGISTER_FWPROXYBUILDER(FWJetProxyBuilder, reco::Jet, "Jets", FWViewType::k3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWJetGlimpseProxyBuilder, reco::Jet, "Jets", FWViewType::kGlimpseBit);
REGISTER_FWPROXYBUILDER(FWJetLegoProxyBuilder, reco::Jet, "Jets", FWViewType::kLegoBit);
