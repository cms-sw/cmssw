// -*- C++ -*-
//
// Package:     Core
// Class  :     GenParticleProxy3DBuilder
// 
/**\class GenParticleProxy3DBuilder GenParticleProxy3DBuilder.h Fireworks/Core/interface/GenParticleProxy3DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: GenParticleProxy3DBuilder.cc,v 1.5 2008/01/27 01:10:42 dmytro Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "RVersion.h"

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "Fireworks/Core/interface/GenParticleProxy3DBuilder.h"


void GenParticleProxy3DBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
    std::cout <<"build called"<<std::endl;

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
     m_item->get(genParticles);
     //fwlite::Handle<reco::TrackCollection> tracks;
     //tracks.getByLabel(*iEvent,"ctfWithMaterialTracks");
     
     if(0 == genParticles ) {
	  std::cout <<"failed to get GenParticles"<<std::endl;
	  return;
     }
     
    
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
      genPart->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(genPart,tlist);
      //cout << it->px()<<" "
      //   <<it->py()<<" "
      //   <<it->pz()<<endl;
      //cout <<" *";
    }
    
}

