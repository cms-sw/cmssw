// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonsRPZProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Dec  4 19:28:07 EST 2008
// $Id: FWMuonsRPZProxyBuilder.cc,v 1.1 2008/12/05 01:51:48 chrjones Exp $
//

// system include files
#include "DataFormats/MuonReco/interface/Muon.h"

// user include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Muons/interface/FWMuonBuilder.h"

class FWMuonsRPZProxyBuilder : public FWRPZ2DSimpleProxyBuilderTemplate<reco::Muon>  {

public:
   FWMuonsRPZProxyBuilder();
   virtual ~FWMuonsRPZProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonsRPZProxyBuilder(const FWMuonsRPZProxyBuilder&); // stop default

   const FWMuonsRPZProxyBuilder& operator=(const FWMuonsRPZProxyBuilder&); // stop default

   // ---------- member data --------------------------------
   void buildRhoPhi(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   void buildRhoZ(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

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
FWMuonsRPZProxyBuilder::FWMuonsRPZProxyBuilder()
{
}

// FWMuonsRPZProxyBuilder::FWMuonsRPZProxyBuilder(const FWMuonsRPZProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWMuonsRPZProxyBuilder::~FWMuonsRPZProxyBuilder()
{
}

//
// assignment operators
//
// const FWMuonsRPZProxyBuilder& FWMuonsRPZProxyBuilder::operator=(const FWMuonsRPZProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWMuonsRPZProxyBuilder temp(rhs);
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
FWMuonsRPZProxyBuilder::buildRhoPhi(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   m_builder.buildMuon(item(), &iData, &oItemHolder, false, false );
}

void
FWMuonsRPZProxyBuilder::buildRhoZ(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   m_builder.buildMuon(item(), &iData, &oItemHolder, true, false );
}

//
// static member functions
//
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWMuonsRPZProxyBuilder,reco::Muon,"Muons");
