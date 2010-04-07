// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGenParticle3DProxyBuilder
//
/**\class FWGenParticle3DProxyBuilder FWGenParticle3DProxyBuilder.h Fireworks/GenParticle/interface/FWGenParticle3DProxyBuilder.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWGenParticle3DProxyBuilder.cc,v 1.7 2010/04/07 14:15:06 amraktad Exp $
// 

#include "TDatabasePDG.h"
#include "TEveTrack.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Candidates/interface/CandidateUtils.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

class FWGenParticle3DProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GenParticle> {

public:
   FWGenParticle3DProxyBuilder();
   virtual ~FWGenParticle3DProxyBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWGenParticle3DProxyBuilder(const FWGenParticle3DProxyBuilder&); // stop default

   const FWGenParticle3DProxyBuilder& operator=(const FWGenParticle3DProxyBuilder&); // stop default
   
   void build(const reco::GenParticle& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   // ---------- member data --------------------------------
   static TDatabasePDG* s_pdg;
};

//______________________________________________________________________________

TDatabasePDG* FWGenParticle3DProxyBuilder::s_pdg = 0;

FWGenParticle3DProxyBuilder::FWGenParticle3DProxyBuilder()
{
}

void
FWGenParticle3DProxyBuilder::build(const reco::GenParticle& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   if (!s_pdg)
      s_pdg = new TDatabasePDG();
 
   char s[1024];
   TParticlePDG* pID = s_pdg->GetParticle(iData.pdgId());
   if ( pID )
      sprintf(s,"gen %s, Pt: %0.1f GeV", pID->GetName(), iData.pt());
   else
      sprintf(s,"gen pdg %d, Pt: %0.1f GeV", iData.pdgId(), iData.pt());

   TEveTrack* trk = fireworks::prepareCandidate( iData, context().getTrackPropagator(), item()->defaultDisplayProperties().color() );    
   trk->MakeTrack();
   trk->SetTitle(s);
   oItemHolder.AddElement( trk );
}

REGISTER_FWPROXYBUILDER(FWGenParticle3DProxyBuilder,reco::GenParticle,"GenParticles", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);

