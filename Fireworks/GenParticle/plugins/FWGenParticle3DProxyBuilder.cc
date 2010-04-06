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
// $Id: FWGenParticle3DProxyBuilder.cc,v 1.5 2010/04/06 20:19:49 amraktad Exp $
// 


#include "TEveTrack.h"
#include "TDatabasePDG.h"
#include "TEveVSDStructs.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"


class TEveTrack;
class TEveTrackPropagator;

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
   TDatabasePDG* m_pdg;
};

//______________________________________________________________________________

FWGenParticle3DProxyBuilder::FWGenParticle3DProxyBuilder()
 :m_pdg(0)
{
   m_pdg = new TDatabasePDG();
}

void
FWGenParticle3DProxyBuilder::build(const reco::GenParticle& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   TEveTrack* trk = fireworks::prepareTrack( iData, context().getTrackPropagator(), item()->defaultDisplayProperties().color() ); 
   
   trk->MakeTrack();
   oItemHolder.AddElement( trk );
   
}

REGISTER_FWPROXYBUILDER(FWGenParticle3DProxyBuilder,reco::GenParticle,"GenParticles", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);

