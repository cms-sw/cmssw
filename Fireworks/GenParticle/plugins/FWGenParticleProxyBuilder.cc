// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGenParticleProxyBuilder
//
/**\class FWGenParticleProxyBuilder 

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWGenParticleProxyBuilder.cc,v 1.7 2010/11/11 20:25:28 amraktad Exp $
// 

#include "TEveTrack.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Candidates/interface/CandidateUtils.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

class FWGenParticleProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GenParticle> {

public:
   FWGenParticleProxyBuilder() {}
   virtual ~FWGenParticleProxyBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWGenParticleProxyBuilder(const FWGenParticleProxyBuilder&); // stop default

   const FWGenParticleProxyBuilder& operator=(const FWGenParticleProxyBuilder&); // stop default
   
   void build(const reco::GenParticle& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*);

};

//______________________________________________________________________________


void
FWGenParticleProxyBuilder::build(const reco::GenParticle& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   TEveTrack* trk = fireworks::prepareCandidate( iData, context().getTrackPropagator() );    
   trk->MakeTrack();
   setupAddElement(trk, &oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWGenParticleProxyBuilder, reco::GenParticle, "GenParticles", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

