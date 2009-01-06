// -*- C++ -*-
//
// Package:     Candidates
// Class  :     FWCandidate3DProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Dec  5 09:56:09 EST 2008
// $Id: FWCandidate3DProxyBuilder.cc,v 1.1 2008/12/05 19:51:27 chrjones Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "DataFormats/Candidate/interface/Candidate.h"

// user include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Candidates/interface/prepareSimpleTrack.h"
#include "Fireworks/Core/interface/FWEventItem.h"

class FWCandidate3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::Candidate>  {
      
public:
   FWCandidate3DProxyBuilder();
   virtual ~FWCandidate3DProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCandidate3DProxyBuilder(const FWCandidate3DProxyBuilder&); // stop default
   
   const FWCandidate3DProxyBuilder& operator=(const FWCandidate3DProxyBuilder&); // stop default

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
FWCandidate3DProxyBuilder::FWCandidate3DProxyBuilder():
m_propagator( new TEveTrackPropagator)
{
   m_propagator->SetMagField( - CmsShowMain::getMagneticField() );
   m_propagator->SetMaxR(123.0);
   m_propagator->SetMaxZ(300.0);
}

// FWCandidate3DProxyBuilder::FWCandidate3DProxyBuilder(const FWCandidate3DProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWCandidate3DProxyBuilder::~FWCandidate3DProxyBuilder()
{
}

//
// assignment operators
//
// const FWCandidate3DProxyBuilder& FWCandidate3DProxyBuilder::operator=(const FWCandidate3DProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWCandidate3DProxyBuilder temp(rhs);
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
FWCandidate3DProxyBuilder::build(const reco::Candidate& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   m_propagator->SetMagField( - CmsShowMain::getMagneticField() );      

   TEveTrack* trk = fireworks::prepareSimpleTrack( iData, m_propagator.get(), &oItemHolder, item()->defaultDisplayProperties().color() );
   trk->MakeTrack();
   oItemHolder.AddElement( trk );
}

//
// static member functions
//
REGISTER_FW3DDATAPROXYBUILDER(FWCandidate3DProxyBuilder,reco::Candidate,"Candidates");
