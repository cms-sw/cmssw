// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWPFTauProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWPFTauProxyBuilder.cc,v 1.2 2010/04/15 13:19:32 yana Exp $
//

// system include files
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Calo/interface/FW3DEveJet.h"
#include "Fireworks/Calo/interface/FWGlimpseEveJet.h"

#include "DataFormats/TauReco/interface/PFTau.h"

class FWPFTauProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFTau>
{
public:
   FWPFTauProxyBuilder() {}
   virtual ~FWPFTauProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFTauProxyBuilder(const FWPFTauProxyBuilder&);    // stop default
   const FWPFTauProxyBuilder& operator=(const FWPFTauProxyBuilder&);    // stop default

   virtual void build( const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWPFTauProxyBuilder::build( const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   const reco::PFTauTagInfo *tauTagInfo = dynamic_cast<const reco::PFTauTagInfo*>((iData.pfTauTagInfoRef().get()));
   const reco::PFJet *jet = dynamic_cast<const reco::PFJet*>((tauTagInfo->pfjetRef().get()));

   FW3DEveJet* cone = new FW3DEveJet( *jet, "jet", "jet");
   cone->SetPickable( kTRUE );
   cone->SetMainTransparency( 75 ); 

   oItemHolder.AddElement( cone );
}

class FWPFTauGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFTau>
{
public:
   FWPFTauGlimpseProxyBuilder() {}
   virtual ~FWPFTauGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFTauGlimpseProxyBuilder(const FWPFTauGlimpseProxyBuilder&);    // stop default
   const FWPFTauGlimpseProxyBuilder& operator=(const FWPFTauGlimpseProxyBuilder&);    // stop default

   virtual void build( const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWPFTauGlimpseProxyBuilder::build( const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   const reco::PFTauTagInfo *tauTagInfo = dynamic_cast<const reco::PFTauTagInfo*>((iData.pfTauTagInfoRef().get()));
   const reco::PFJet *jet = dynamic_cast<const reco::PFJet*>((tauTagInfo->pfjetRef().get()));

   FWGlimpseEveJet* cone = new FWGlimpseEveJet( jet, "jet", "jet");
   cone->SetPickable( kTRUE );
   cone->SetMainTransparency( 50 ); 
   cone->SetDrawConeCap( kFALSE );

   oItemHolder.AddElement( cone );
}

class FWPFTauLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFTau>
{
public:
   FWPFTauLegoProxyBuilder() {}
   virtual ~FWPFTauLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFTauLegoProxyBuilder(const FWPFTauLegoProxyBuilder&);    // stop default
   const FWPFTauLegoProxyBuilder& operator=(const FWPFTauLegoProxyBuilder&);    // stop default

   virtual void build( const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWPFTauLegoProxyBuilder::build( const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   const unsigned int nLineSegments = 20;
   const double jetRadius = 0.17;   //10 degree
   TEveStraightLineSet* container = new TEveStraightLineSet();
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

REGISTER_FWPROXYBUILDER(FWPFTauProxyBuilder, reco::PFTau, "PFTau", FWViewType::k3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWPFTauGlimpseProxyBuilder, reco::PFTau, "PFTau", FWViewType::kGlimpseBit);
REGISTER_FWPROXYBUILDER(FWPFTauLegoProxyBuilder, reco::PFTau, "PFTau", FWViewType::kLegoBit);
