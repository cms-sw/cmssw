// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Dec  4 19:28:07 EST 2008
// $Id: FWMuonProxyBuilder.cc,v 1.13 2010/08/19 13:39:17 yana Exp $
//

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Muons/interface/FWMuonBuilder.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class FWMuonProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>
{
public:
   FWMuonProxyBuilder( void ) {}
   virtual ~FWMuonProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWMuonProxyBuilder( const FWMuonProxyBuilder& );
   // Disable default assignment operator
   const FWMuonProxyBuilder& operator=( const FWMuonProxyBuilder& );


   virtual void build( const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );

   virtual void localModelChanges( const FWModelId& iId, TEveElement* iCompound,
                                   FWViewType::EType viewType, const FWViewContext* vc );

   mutable FWMuonBuilder m_builder;
};

void
FWMuonProxyBuilder::build( const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
   m_builder.buildMuon( this, &iData, &oItemHolder, true, false );

   increaseComponentTransparency( iIndex, &oItemHolder, "Chamber", 60 );
}

void
FWMuonProxyBuilder::localModelChanges( const FWModelId& iId, TEveElement* iCompound,
                                       FWViewType::EType viewType, const FWViewContext* vc )
{
   increaseComponentTransparency( iId.index(), iCompound, "Chamber", 60 );
}

REGISTER_FWPROXYBUILDER( FWMuonProxyBuilder, reco::Muon, "Muons", FWViewType::kAll3DBits | FWViewType::kRhoZBit );
