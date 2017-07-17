// -*- C++ -*-
//
// Package:     CandidatePtrs
// Class  :     FWCandidatePtrProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Dec  5 09:56:09 EST 2008
//

#include "TEveTrack.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"

#include "DataFormats/Candidate/interface/Candidate.h"

class FWCandidatePtrProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::CandidatePtr>  {
      
public:
   FWCandidatePtrProxyBuilder() {}
   virtual ~FWCandidatePtrProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCandidatePtrProxyBuilder(const FWCandidatePtrProxyBuilder&); // stop default
   const FWCandidatePtrProxyBuilder& operator=(const FWCandidatePtrProxyBuilder&); // stop default

   using FWSimpleProxyBuilderTemplate<reco::CandidatePtr>::build;
   void build(const reco::CandidatePtr& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) override;
};


void 
FWCandidatePtrProxyBuilder::build(const reco::CandidatePtr& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   TEveTrack* trk = fireworks::prepareCandidate( *iData, context().getTrackPropagator() ); 
   
   trk->MakeTrack();
   setupAddElement(trk, &oItemHolder);
   {
      TEveRecTrack t;
      t.fBeta = 1.;
      t.fV = TEveVector(iData->vx(),iData->vy(),iData->vz());
      t.fP = TEveVector(-iData->p4().px(), -iData->p4().py(), -iData->p4().pz());
      t.fSign = iData->charge();
      TEveTrack* trk2= new TEveTrack(&t, context().getTrackPropagator());
      trk2->SetLineStyle(7);
      trk2->MakeTrack();
      setupAddElement(trk2, &oItemHolder);

   }

}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWCandidatePtrProxyBuilder, reco::CandidatePtr, "CandidatePtrs", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
