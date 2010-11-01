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
// $Id: FWGenParticleRPZProxyBuilder.cc,v 1.3 2010/01/22 19:52:57 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveVSDStructs.h"
#include "RVersion.h"
#include "TDatabasePDG.h"
#include "TEveVSDStructs.h"

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"



class FWGenParticleRPZProxyBuilder : public FWRPZDataProxyBuilder {

public:
   FWGenParticleRPZProxyBuilder();
   virtual ~FWGenParticleRPZProxyBuilder() {
   }

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   FWGenParticleRPZProxyBuilder(const FWGenParticleRPZProxyBuilder&);    // stop default

   const FWGenParticleRPZProxyBuilder& operator=(const FWGenParticleRPZProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
   TDatabasePDG* m_pdg;
};

FWGenParticleRPZProxyBuilder::FWGenParticleRPZProxyBuilder()
{
   m_pdg = new TDatabasePDG();
}

void FWGenParticleRPZProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   reco::GenParticleCollection const * genParticles=0;
   iItem->get(genParticles);
   if(0 == genParticles ) return;

  if(0==*product) {
      *product = new TEveElementList();
   } else {
      (*product)->DestroyElements();
   }
   TEveElementList* tlist = *product;

   int index=0;
   TEveRecTrack t;
   t.fBeta = 1.;
   reco::GenParticleCollection::const_iterator it = genParticles->begin(),
      end = genParticles->end();
   for( ; it != end; ++it,++index) {
      t.fP = TEveVector(it->px(),
                        it->py(),
                        it->pz());
      t.fV = TEveVector(it->vx(),
                        it->vy(),
                        it->vz());
      t.fSign = it->charge();

      TEveTrack* track = new TEveTrack(&t, context().getTrackPropagator());
     
      char s[1024];
      TParticlePDG* pID = m_pdg->GetParticle(it->pdgId());
      if ( pID )
         sprintf(s,"gen %s, Pt: %0.1f GeV", pID->GetName(), it->pt());
      else
         sprintf(s,"gen pdg %d, Pt: %0.1f GeV", it->pdgId(), it->pt());

      track->SetMainColor(iItem->defaultDisplayProperties().color());
      track->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
      track->SetTitle(s);
      track->MakeTrack();
      tlist->AddElement(track);
   }
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWGenParticleRPZProxyBuilder,reco::GenParticleCollection,"GenParticles");

