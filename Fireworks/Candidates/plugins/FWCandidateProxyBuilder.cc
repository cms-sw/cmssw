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
// $Id: FWCandidateProxyBuilder.cc,v 1.4 2010/01/25 20:55:05 amraktad Exp $
//

#include "TEveTrack.h"
#include "TEveVSDStructs.h"
#include "Fireworks/Core/interface/Context.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/Candidate/interface/Candidate.h"

class Track;
class TEveTrackPropagator;

class FWCandidateProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Candidate>  {
      
public:
   FWCandidateProxyBuilder() {}
   virtual ~FWCandidateProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCandidateProxyBuilder(const FWCandidateProxyBuilder&); // stop default
   const FWCandidateProxyBuilder& operator=(const FWCandidateProxyBuilder&); // stop default

   void build(const reco::Candidate& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
};


void 
FWCandidateProxyBuilder::build(const reco::Candidate& iData, unsigned int iIndex,TEveElement& oItemHolder) const
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
REGISTER_FWPROXYBUILDER(FWCandidateProxyBuilder,reco::Candidate,"Candidates", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
