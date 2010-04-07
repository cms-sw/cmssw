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
// $Id: FWCandidateProxyBuilder.cc,v 1.1 2010/04/07 08:05:46 yana Exp $
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

   void build(const reco::Candidate& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};


void 
FWCandidateProxyBuilder::build(const reco::Candidate& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   TEveTrack* trk = fireworks::prepareCandidate( iData, context().getTrackPropagator(), item()->defaultDisplayProperties().color() ); 
   
   trk->MakeTrack();
   oItemHolder.AddElement( trk );
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWCandidateProxyBuilder, reco::Candidate, "Candidates", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
