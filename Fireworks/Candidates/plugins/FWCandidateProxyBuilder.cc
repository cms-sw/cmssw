// -*- C++ -*-
//
// Package:     Candidates
// Class  :     FWCandidateProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Dec  5 09:56:09 EST 2008
// $Id: FWCandidateProxyBuilder.cc,v 1.5 2010/04/16 11:28:03 amraktad Exp $
//

#include "TEveTrack.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Candidates/interface/CandidateUtils.h"

#include "DataFormats/Candidate/interface/Candidate.h"

class FWCandidateProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Candidate>  {
      
public:
   FWCandidateProxyBuilder() {}
   virtual ~FWCandidateProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCandidateProxyBuilder(const FWCandidateProxyBuilder&); // stop default
   const FWCandidateProxyBuilder& operator=(const FWCandidateProxyBuilder&); // stop default

   void build(const reco::Candidate& iData, unsigned int iIndex, TEveElement& oItemHolder);
};


void 
FWCandidateProxyBuilder::build(const reco::Candidate& iData, unsigned int iIndex, TEveElement& oItemHolder) 
{
   TEveTrack* trk = fireworks::prepareCandidate( iData, context().getTrackPropagator() ); 
   
   trk->MakeTrack();
   setupAddElement(trk, &oItemHolder);
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWCandidateProxyBuilder, reco::Candidate, "Candidates", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
