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
// $Id: FWCandidateRPZProxyBuilder.cc,v 1.3 2010/01/21 21:02:11 amraktad Exp $
//
#include "TEveVSDStructs.h"
#include "TEveTrack.h"

// user include files
#include "Fireworks/Core/interface/FWRPZSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"

#include "DataFormats/Candidate/interface/Candidate.h"

class TEveTrack;
class TEveTrackPropagator;

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

};

FWCandidateRPZProxyBuilder::FWCandidateRPZProxyBuilder()
{
}

FWCandidateRPZProxyBuilder::~FWCandidateRPZProxyBuilder()
{
}

void 
FWCandidateRPZProxyBuilder::build(const reco::Candidate& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   TEveRecTrack t;
   t.fBeta = 1.;
   t.fP = TEveVector( iData.px(), iData.py(), iData.pz() );
   t.fV = TEveVector( iData.vertex().x(), iData.vertex().y(), iData.vertex().z() );
   t.fSign = iData.charge();
   TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator());
   trk->SetMainColor( item()->defaultDisplayProperties().color());
   trk->MakeTrack();
   oItemHolder.AddElement( trk );
}

//
// static member functions
//
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWCandidateRPZProxyBuilder,reco::Candidate,"Candidates");
