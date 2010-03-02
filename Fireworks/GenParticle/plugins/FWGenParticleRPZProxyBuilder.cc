// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGenParticleRPZProxyBuilder
//
/**\class FWGenParticleRPZProxyBuilder FWGenParticleRPZProxyBuilder.h Fireworks/Core/interface/FWGenParticleRPZProxyBuilder.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWGenParticleRPZProxyBuilder.cc,v 1.4 2010/01/22 20:56:59 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveVSDStructs.h"
#include "RVersion.h"
#include "TDatabasePDG.h"
#include "TEveVSDStructs.h"

// user include files
#include "Fireworks/Core/interface/FWRPZSimpleProxyBuilderTemplate.h"

#include "Fireworks/Core/interface/Context.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"



class FWGenParticleRPZProxyBuilder : public FWRPZSimpleProxyBuilderTemplate<reco::GenParticle> {

public:
   FWGenParticleRPZProxyBuilder();
   virtual ~FWGenParticleRPZProxyBuilder() {
   }

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const reco::GenParticle& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   FWGenParticleRPZProxyBuilder(const FWGenParticleRPZProxyBuilder&);    // stop default

   const FWGenParticleRPZProxyBuilder& operator=(const FWGenParticleRPZProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
   TDatabasePDG* m_pdg;
};

FWGenParticleRPZProxyBuilder::FWGenParticleRPZProxyBuilder()
{
   m_pdg = new TDatabasePDG();
}

void FWGenParticleRPZProxyBuilder::build(const reco::GenParticle& iData, unsigned int iIndex,TEveElement& oItemHolder) const

{
   TEveRecTrack t;
   t.fBeta = 1.;
   t.fP = TEveVector(iData.px(),
                     iData.py(),
                     iData.pz());
   t.fV = TEveVector(iData.vx(),
                     iData.vy(),
                     iData.vz());
   t.fSign = iData.charge();

   TEveTrack* track = new TEveTrack(&t, context().getTrackPropagator());
   
   char s[1024];
   TParticlePDG* pID = m_pdg->GetParticle(iData.pdgId());
   if ( pID )
      sprintf(s,"gen %s, Pt: %0.1f GeV", pID->GetName(), iData.pt());
   else
      sprintf(s,"gen pdg %d, Pt: %0.1f GeV", iData.pdgId(), iData.pt());
   
   oItemHolder.SetElementTitle(s);
   track->MakeTrack();
   oItemHolder.AddElement(track);
}


REGISTER_FWRPZDATAPROXYBUILDERBASE(FWGenParticleRPZProxyBuilder,reco::GenParticle,"GenParticles");

