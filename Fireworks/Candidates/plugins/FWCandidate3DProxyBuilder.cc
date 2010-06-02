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
// $Id: FWCandidate3DProxyBuilder.cc,v 1.3 2010/01/21 21:02:11 amraktad Exp $
//

#include "TEveTrack.h"
#include "TEveVSDStructs.h"
#include "Fireworks/Core/interface/Context.h"

// user include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/Candidate/interface/Candidate.h"

class Track;
class TEveTrackPropagator;

class FWCandidate3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::Candidate>  {
      
public:
   FWCandidate3DProxyBuilder() {}
   virtual ~FWCandidate3DProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCandidate3DProxyBuilder(const FWCandidate3DProxyBuilder&); // stop default
   const FWCandidate3DProxyBuilder& operator=(const FWCandidate3DProxyBuilder&); // stop default

   void build(const reco::Candidate& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
};


void 
FWCandidate3DProxyBuilder::build(const reco::Candidate& iData, unsigned int iIndex,TEveElement& oItemHolder) const
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
REGISTER_FW3DDATAPROXYBUILDER(FWCandidate3DProxyBuilder,reco::Candidate,"Candidates");
