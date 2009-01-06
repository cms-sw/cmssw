// -*- C++ -*-
//
// Package:     Candidates
// Class  :     FWCandidateRPZProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Dec  5 09:54:40 EST 2008
// $Id: FWCandidateRPZProxyBuilder.cc,v 1.1 2008/12/05 19:51:27 chrjones Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "DataFormats/Candidate/interface/Candidate.h"

// user include files
#include "Fireworks/Core/interface/FWRPZSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Candidates/interface/prepareSimpleTrack.h"
#include "Fireworks/Core/interface/FWEventItem.h"

class FWCandidateRPZProxyBuilder : public FWRPZSimpleProxyBuilderTemplate<reco::Candidate>  {
      
public:
   FWCandidateRPZProxyBuilder();
   virtual ~FWCandidateRPZProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWCandidateRPZProxyBuilder(const FWCandidateRPZProxyBuilder&); // stop default
   
   const FWCandidateRPZProxyBuilder& operator=(const FWCandidateRPZProxyBuilder&); // stop default
   
   void build(const reco::Candidate& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   // ---------- member data --------------------------------

   FWEvePtr<TEveTrackPropagator> m_propagator;
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
FWCandidateRPZProxyBuilder::FWCandidateRPZProxyBuilder():
m_propagator( new TEveTrackPropagator)
{
   m_propagator->SetMagField( - CmsShowMain::getMagneticField() );
   m_propagator->SetMaxR(123.0);
   m_propagator->SetMaxZ(300.0);
}

// FWCandidateRPZProxyBuilder::FWCandidateRPZProxyBuilder(const FWCandidateRPZProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWCandidateRPZProxyBuilder::~FWCandidateRPZProxyBuilder()
{
}

//
// assignment operators
//
// const FWCandidateRPZProxyBuilder& FWCandidateRPZProxyBuilder::operator=(const FWCandidateRPZProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWCandidateRPZProxyBuilder temp(rhs);
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
FWCandidateRPZProxyBuilder::build(const reco::Candidate& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   m_propagator->SetMagField( - CmsShowMain::getMagneticField() );      
   
   TEveTrack* trk = fireworks::prepareSimpleTrack( iData, m_propagator.get(), &oItemHolder, item()->defaultDisplayProperties().color() );
   trk->MakeTrack();
   oItemHolder.AddElement( trk );
}

//
// static member functions
//
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWCandidateRPZProxyBuilder,reco::Candidate,"Candidates");
