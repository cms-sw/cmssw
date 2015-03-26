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
//

#include "TEveTrack.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"


class FWCandidateProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Candidate>  {
      
public:
   FWCandidateProxyBuilder() {}
   virtual ~FWCandidateProxyBuilder() {}

   virtual void setItem(const FWEventItem* iItem) override
   {
      FWProxyBuilderBase::setItem(iItem);
      if (iItem)
      {
         iItem->getConfig()->assertParam("Draw backward extrapolation", false);
      }
   }

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCandidateProxyBuilder(const FWCandidateProxyBuilder&); // stop default
   const FWCandidateProxyBuilder& operator=(const FWCandidateProxyBuilder&); // stop default

   using FWSimpleProxyBuilderTemplate<reco::Candidate>::build;
   void build(const reco::Candidate& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) override;
};


void 
FWCandidateProxyBuilder::build(const reco::Candidate& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   TEveTrack* trk = fireworks::prepareCandidate( iData, context().getTrackPropagator() ); 
   
   trk->MakeTrack();
   setupAddElement(trk, &oItemHolder);

   if ( item()->getConfig()->value<bool>("Draw backward extrapolation"))
   {
      TEveRecTrack t;
      t.fBeta = 1.;
      t.fV = TEveVector(iData.vx(),iData.vy(),iData.vz());
      t.fP = TEveVector(-iData.p4().px(), -iData.p4().py(), -iData.p4().pz());
      t.fSign = iData.charge();
      TEveTrack* trk2= new TEveTrack(&t, context().getTrackPropagator());
      trk2->SetLineStyle(7);
      trk2->MakeTrack();
      setupAddElement(trk2, &oItemHolder);

   }

}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWCandidateProxyBuilder, reco::Candidate, "Candidates", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
