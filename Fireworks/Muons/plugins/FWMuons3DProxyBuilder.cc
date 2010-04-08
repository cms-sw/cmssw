// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuons3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Dec  4 19:28:07 EST 2008
// $Id: FWMuons3DProxyBuilder.cc,v 1.2 2009/01/23 21:35:46 amraktad Exp $
//

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Muons/interface/FWMuonBuilder.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class FWMuons3DProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuons3DProxyBuilder() {}
   virtual ~FWMuons3DProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuons3DProxyBuilder(const FWMuons3DProxyBuilder&); // stop default

   const FWMuons3DProxyBuilder& operator=(const FWMuons3DProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void build(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   mutable FWMuonBuilder m_builder;
};

void
FWMuons3DProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   m_builder.buildMuon(item(), &iData, &oItemHolder, true, false);
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWMuons3DProxyBuilder, reco::Muon, "Muons", FWViewType::k3DBit | FWViewType::kRhoPhiBit | FWViewType::kRhoZBit);
