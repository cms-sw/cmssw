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
// $Id: FWCandidateRPZProxyBuilder.cc,v 1.2 2009/01/06 20:07:48 chrjones Exp $
//

// user include files
#include "Fireworks/Core/interface/FWRPZSimpleProxyBuilderTemplate.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Core/interface/FWEventItem.h"

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
   TEveTrack* trk = fireworks::prepareTrack( iData, context().getTrackPropagator(), item()->defaultDisplayProperties().color() );
   trk->MakeTrack();
   oItemHolder.AddElement( trk );
}

//
// static member functions
//
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWCandidateRPZProxyBuilder,reco::Candidate,"Candidates");
