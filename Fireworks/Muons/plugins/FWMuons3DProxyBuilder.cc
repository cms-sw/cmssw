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
// $Id: FWMuons3DProxyBuilder.cc,v 1.1 2008/12/05 01:51:48 chrjones Exp $
//

// system include files
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

// user include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Muons/interface/FWMuonBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

class FWMuons3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuons3DProxyBuilder();
   virtual ~FWMuons3DProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuons3DProxyBuilder(const FWMuons3DProxyBuilder&); // stop default

   const FWMuons3DProxyBuilder& operator=(const FWMuons3DProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void build(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   mutable FWMuonBuilder m_builder;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWMuons3DProxyBuilder::FWMuons3DProxyBuilder()
{
}

// FWMuons3DProxyBuilder::FWMuons3DProxyBuilder(const FWMuons3DProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWMuons3DProxyBuilder::~FWMuons3DProxyBuilder()
{
}

//
// assignment operators
//
// const FWMuons3DProxyBuilder& FWMuons3DProxyBuilder::operator=(const FWMuons3DProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWMuons3DProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

void
FWMuons3DProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   m_builder.buildMuon(item(), &iData, &oItemHolder, true, false );
}

//
// static member functions
//
REGISTER_FW3DDATAPROXYBUILDER(FWMuons3DProxyBuilder,reco::Muon,"Muons");
