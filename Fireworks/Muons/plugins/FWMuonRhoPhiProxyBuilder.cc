// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonRhoPhiProxyBuilder
//
// $Id: FWMuonRhoPhiProxyBuilder.cc,v 1.2 2010/11/11 20:25:28 amraktad Exp $
//

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Muons/interface/FWMuonBuilder.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class FWMuonRhoPhiProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>
{
public:
   FWMuonRhoPhiProxyBuilder( void ) {}
   virtual ~FWMuonRhoPhiProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWMuonRhoPhiProxyBuilder( const FWMuonRhoPhiProxyBuilder& );
   // Disable default assignment operator
   const FWMuonRhoPhiProxyBuilder& operator=( const FWMuonRhoPhiProxyBuilder& );

   void build( const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );

   virtual void localModelChanges( const FWModelId& iId, TEveElement* iCompound,
                                   FWViewType::EType viewType, const FWViewContext* vc );

   mutable FWMuonBuilder m_builder;
};

void
FWMuonRhoPhiProxyBuilder::build( const reco::Muon& iData, unsigned int iIndex,
                                 TEveElement& oItemHolder, const FWViewContext* ) 
{
   // To build in RhoPhi we should simply disable the Endcap drawing
   // by passing a false flag to a muon builder:
   m_builder.buildMuon( this, &iData, &oItemHolder, false, false );

   increaseComponentTransparency( iIndex, &oItemHolder, "Chamber", 40 );
}

void
FWMuonRhoPhiProxyBuilder::localModelChanges( const FWModelId& iId, TEveElement* iCompound,
                                             FWViewType::EType viewType, const FWViewContext* vc )
{
   increaseComponentTransparency( iId.index(), iCompound, "Chamber", 40 );
}

REGISTER_FWPROXYBUILDER( FWMuonRhoPhiProxyBuilder, reco::Muon, "Muons", FWViewType::kRhoPhiBit |  FWViewType::kRhoPhiPFBit);
