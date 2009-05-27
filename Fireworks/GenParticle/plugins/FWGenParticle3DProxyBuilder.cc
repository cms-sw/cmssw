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
// $Id: FWGenParticle3DProxyBuilder.cc,v 1.2 2009/05/26 21:35:46 yanjuntu Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "RVersion.h"
#include "TDatabasePDG.h"

// user include files


#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"



class FWGenParticle3DProxyBuilder : public FW3DDataProxyBuilder {

public:
   FWGenParticle3DProxyBuilder();
   virtual ~FWGenParticle3DProxyBuilder() {
   }

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   FWGenParticle3DProxyBuilder(const FWGenParticle3DProxyBuilder&);    // stop default

   const FWGenParticle3DProxyBuilder& operator=(const FWGenParticle3DProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
   TDatabasePDG* m_pdg;
};

FWGenParticle3DProxyBuilder::FWGenParticle3DProxyBuilder()
{
   m_pdg = new TDatabasePDG();
}

void FWGenParticle3DProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   //since we created it, we know the type (would like to do this better)
   TEveTrackList* tlist = dynamic_cast<TEveTrackList*>(*product);
   if ( !tlist && *product ) {
      std::cout << "incorrect type" << std::endl;
      return;
   }

   if(0 == tlist) {
      tlist =  new TEveTrackList(iItem->name().c_str());
      *product = tlist;
      tlist->SetMainColor(iItem->defaultDisplayProperties().color());
      TEveTrackPropagator* rnrStyle = tlist->GetPropagator();
      //units are Tesla
      rnrStyle->SetMagField( -4.0);
      //get this from geometry, units are CM
      rnrStyle->SetMaxR(120.0);
      rnrStyle->SetMaxZ(300.0);

      gEve->AddElement(tlist);
   } else {
      tlist->DestroyElements();
   }


   reco::GenParticleCollection const * genParticles=0;
   iItem->get(genParticles);
   //fwlite::Handle<reco::TrackCollection> tracks;
   //tracks.getByLabel(*iEvent,"ctfWithMaterialTracks");

   if(0 == genParticles ) return;

   TEveTrackPropagator* rnrStyle = tlist->GetPropagator();

   int index=0;
   //cout <<"----"<<endl;
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
      TEveTrack* genPart = new TEveTrack(&t,rnrStyle);
     
      char s[1024];
      TParticlePDG* pID = m_pdg->GetParticle(it->pdgId());
      if ( pID )
         sprintf(s,"gen %s, Pt: %0.1f GeV", pID->GetName(), it->pt());
      else
         sprintf(s,"gen pdg %d, Pt: %0.1f GeV", it->pdgId(), it->pt());
      genPart->SetTitle(s);
      genPart->SetMainColor(iItem->defaultDisplayProperties().color());
      genPart->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
      genPart->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
     
      genPart->MakeTrack();
      tlist->AddElement(genPart);
     
      // std::cout << it->px()<<" "
// 	   <<it->py()<<" "
// 	   <<it->pz()<<std::endl;
//       std::cout <<" *";
   }

}

REGISTER_FW3DDATAPROXYBUILDER(FWGenParticle3DProxyBuilder,reco::GenParticleCollection,"GenParticles");

