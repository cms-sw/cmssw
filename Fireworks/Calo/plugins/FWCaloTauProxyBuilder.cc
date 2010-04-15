// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTauProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWCaloTauProxyBuilder.cc,v 1.1 2010/04/15 12:45:02 yana Exp $
//

// system include files
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Calo/interface/FW3DEveJet.h"
#include "Fireworks/Calo/interface/FWGlimpseEveJet.h"

#include "DataFormats/TauReco/interface/CaloTau.h"

class FWCaloTauProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::CaloTau>
{
public:
   FWCaloTauProxyBuilder() {}
   virtual ~FWCaloTauProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloTauProxyBuilder(const FWCaloTauProxyBuilder&);    // stop default
   const FWCaloTauProxyBuilder& operator=(const FWCaloTauProxyBuilder&);    // stop default

   virtual void build( const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWCaloTauProxyBuilder::build( const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   const reco::CaloTauTagInfo *tauTagInfo = dynamic_cast<const reco::CaloTauTagInfo*>((iData.caloTauTagInfoRef().get()));
   const reco::CaloJet *jet = dynamic_cast<const reco::CaloJet*>((tauTagInfo->calojetRef().get()));

   FW3DEveJet* cone = new FW3DEveJet( *jet, "jet", "jet");
   cone->SetPickable( kTRUE );
   cone->SetMainTransparency( 75 ); 

   oItemHolder.AddElement( cone );
}

class FWCaloTauGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::CaloTau>
{
public:
   FWCaloTauGlimpseProxyBuilder() {}
   virtual ~FWCaloTauGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloTauGlimpseProxyBuilder(const FWCaloTauGlimpseProxyBuilder&);    // stop default
   const FWCaloTauGlimpseProxyBuilder& operator=(const FWCaloTauGlimpseProxyBuilder&);    // stop default

   virtual void build( const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWCaloTauGlimpseProxyBuilder::build( const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   const reco::CaloTauTagInfo *tauTagInfo = dynamic_cast<const reco::CaloTauTagInfo*>((iData.caloTauTagInfoRef().get()));
   const reco::CaloJet *jet = dynamic_cast<const reco::CaloJet*>((tauTagInfo->calojetRef().get()));

   FWGlimpseEveJet* cone = new FWGlimpseEveJet( jet, "jet", "jet");
   cone->SetPickable( kTRUE );
   cone->SetMainTransparency( 50 ); 
   cone->SetDrawConeCap( kFALSE );

   oItemHolder.AddElement( cone );
}

class FWCaloTauLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::CaloTau>
{
public:
   FWCaloTauLegoProxyBuilder() {}
   virtual ~FWCaloTauLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloTauLegoProxyBuilder(const FWCaloTauLegoProxyBuilder&);    // stop default
   const FWCaloTauLegoProxyBuilder& operator=(const FWCaloTauLegoProxyBuilder&);    // stop default

   virtual void build( const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWCaloTauLegoProxyBuilder::build( const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
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

REGISTER_FWPROXYBUILDER(FWCaloTauProxyBuilder, reco::CaloTau, "CaloTau", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWCaloTauGlimpseProxyBuilder, reco::CaloTau, "CaloTau", FWViewType::kGlimpseBit);
REGISTER_FWPROXYBUILDER(FWCaloTauLegoProxyBuilder, reco::CaloTau, "CaloTau", FWViewType::kLegoBit);
