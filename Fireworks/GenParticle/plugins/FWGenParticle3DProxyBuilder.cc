// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGenParticle3DProxyBuilder
//
/**\class FWGenParticle3DProxyBuilder FWGenParticle3DProxyBuilder.h Fireworks/Core/interface/FWGenParticle3DProxyBuilder.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWGenParticle3DProxyBuilder.cc,v 1.4 2010/03/02 20:23:20 chrjones Exp $
// 


#include "TEveTrack.h"
#include "TDatabasePDG.h"
#include "TEveVSDStructs.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"


class TEveTrack;
class TEveTrackPropagator;

class FWGenParticle3DProxyBuilder : public  FWProxyBuilderBase {

public:
   FWGenParticle3DProxyBuilder();
   virtual ~FWGenParticle3DProxyBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWGenParticle3DProxyBuilder(const FWGenParticle3DProxyBuilder&); // stop default

   const FWGenParticle3DProxyBuilder& operator=(const FWGenParticle3DProxyBuilder&); // stop default

   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product);

   // ---------- member data --------------------------------
   TDatabasePDG* m_pdg;
};

//______________________________________________________________________________

FWGenParticle3DProxyBuilder::FWGenParticle3DProxyBuilder()
 :m_pdg(0)
{
   m_pdg = new TDatabasePDG();
}

void FWGenParticle3DProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{ 
   reco::GenParticleCollection const * genParticles=0;
   iItem->get(genParticles);
   if(0 == genParticles ) return;

   (*product)->DestroyElements();
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
      tlist->ProjectChild(track);
   }
}

REGISTER_FWPROXYBUILDER(FWGenParticle3DProxyBuilder,reco::GenParticleCollection,"GenParticles", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);

