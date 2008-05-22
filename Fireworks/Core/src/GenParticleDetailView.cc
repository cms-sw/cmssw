#include "TEveElement.h"
#include "TEveGeoNode.h"
#include "TEveManager.h"
#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/GenParticleDetailView.h"

GenParticleDetailView::GenParticleDetailView () 
{

}

GenParticleDetailView::~GenParticleDetailView ()
{

}

void GenParticleDetailView::build (TEveElementList **product, const FWModelId &id)
{
     m_item = id.item();
     // printf("calling ElectronDetailView::buildRhoZ\n");
     TEveElementList* tList = *product;
     if(0 == tList) {
	  tList =  new TEveElementList(m_item->name().c_str(),"Supercluster RhoZ",true);
	  *product = tList;
	  tList->SetMainColor(m_item->defaultDisplayProperties().color());
     } else {
	  return;
// 	  tList->DestroyElements();
     }

     TEveTrackPropagator* rnrStyle = new TEveTrackPropagator();
     //units are Tesla
     rnrStyle->SetMagField( -4.0);
     //get this from geometry, units are CM
     rnrStyle->SetMaxR(120.0);
     rnrStyle->SetMaxZ(300.0);    
     
     reco::GenParticleCollection const * genParticles=0;
     m_item->get(genParticles);
     //fwlite::Handle<reco::TrackCollection> tracks;
     //tracks.getByLabel(*iEvent,"ctfWithMaterialTracks");
     
     if(0 == genParticles ) {
	  std::cout <<"failed to get GenParticles"<<std::endl;
	  return;
     }
     
     //  Original Commented out here
     //  TEveTrackPropagator* rnrStyle = tList->GetPropagator();
     
     int index=0;
     //cout <<"----"<<endl;
     TEveRecTrack t;
     
     t.fBeta = 1.;
     reco::GenParticleCollection::const_iterator it = genParticles->begin(),
       end = genParticles->end();
     for( ; it != end; ++it,++index) {
	  if (index != id.index())
	       continue;
	  t.fP = TEveVector(it->px(),
			    it->py(),
			    it->pz());
	  t.fV = TEveVector(it->vx(),
			    it->vy(),
			    it->vz());
	  t.fSign = it->charge();
	  
	  TEveElementList* genPartList = new TEveElementList(Form("genParticle%d",index));
	  gEve->AddElement(genPartList,tList);  
	  TEveTrack* genPart = new TEveTrack(&t,rnrStyle);
	  genPart->SetMainColor(m_item->defaultDisplayProperties().color());
	  genPart->MakeTrack();
	  genPartList->AddElement(genPart);
    
     }
}
