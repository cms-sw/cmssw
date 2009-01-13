// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGenParticleProxy3DBuilder
//
/**\class FWGenParticleProxy3DBuilder FWGenParticleProxy3DBuilder.h Fireworks/Core/interface/FWGenParticleProxy3DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWGenParticleProxy3DBuilder.cc,v 1.9 2008/11/26 16:19:15 chrjones Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "RVersion.h"
#include "TDatabasePDG.h"

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"



class FWGenParticleProxy3DBuilder : public FWRPZDataProxyBuilder {
      
   public:
      FWGenParticleProxy3DBuilder();
      virtual ~FWGenParticleProxy3DBuilder() {}
      
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      REGISTER_PROXYBUILDER_METHODS();
      
   private:
      virtual void build(const FWEventItem* iItem, TEveElementList** product);
      
      FWGenParticleProxy3DBuilder(const FWGenParticleProxy3DBuilder&); // stop default
      
      const FWGenParticleProxy3DBuilder& operator=(const FWGenParticleProxy3DBuilder&); // stop default
      
      // ---------- member data --------------------------------
      TDatabasePDG* m_pdg;
};

FWGenParticleProxy3DBuilder::FWGenParticleProxy3DBuilder()
{
  m_pdg = new TDatabasePDG();
}

void FWGenParticleProxy3DBuilder::build(const FWEventItem* iItem, TEveElementList** product)
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
      gEve->AddElement(genPart,tlist);
      //cout << it->px()<<" "
      //   <<it->py()<<" "
      //   <<it->pz()<<endl;
      //cout <<" *";
    }

}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWGenParticleProxy3DBuilder,reco::GenParticleCollection,"GenParticles");

