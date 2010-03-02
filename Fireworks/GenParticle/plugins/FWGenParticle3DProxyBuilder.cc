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
// $Id: FWGenParticle3DProxyBuilder.cc,v 1.3 2010/01/22 20:56:59 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "RVersion.h"
#include "TDatabasePDG.h"
#include "TEveVSDStructs.h"
#include "TRandom.h"
#include "TEveStraightLineSet.h"
#include "TEveVSDStructs.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"


class TEveTrack;
class TEveTrackPropagator;

class FWGenParticle3DProxyBuilder : public  FW3DSimpleProxyBuilderTemplate<reco::GenParticle> {

public:
   FWGenParticle3DProxyBuilder();
   virtual ~FWGenParticle3DProxyBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWGenParticle3DProxyBuilder(const FWGenParticle3DProxyBuilder&); // stop default

   const FWGenParticle3DProxyBuilder& operator=(const FWGenParticle3DProxyBuilder&); // stop default

   virtual void build(const reco::GenParticle& iData,
                      unsigned int iIndex,
                      TEveElement& oItheHolder) const;

   // ---------- member data --------------------------------
   TDatabasePDG* m_pdg;
};

//______________________________________________________________________________

FWGenParticle3DProxyBuilder::FWGenParticle3DProxyBuilder()
 :m_pdg(0)
{
   m_pdg = new TDatabasePDG();
}

void FWGenParticle3DProxyBuilder::build(const reco::GenParticle& iData,
                                        unsigned int iIndex,
                                        TEveElement& oItemHolder) const
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
   
   track->SetTitle(s);
   oItemHolder.SetElementTitle(s);
   track->MakeTrack();
   oItemHolder.AddElement(track);
}

REGISTER_FW3DDATAPROXYBUILDER(FWGenParticle3DProxyBuilder,reco::GenParticle,"GenParticles");

