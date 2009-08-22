#include "TEveScene.h"
#include "TGLViewer.h"
#include "TGFrame.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "Fireworks/Core/interface/FWDetailView.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"

class FWGenParticleDetailView : public FWDetailView<reco::GenParticle> {

public:
   FWGenParticleDetailView();
   virtual ~FWGenParticleDetailView();

   virtual void build (const FWModelId &id, const reco::GenParticle*, TEveWindowSlot*);

protected:
   void setItem (const FWEventItem *iItem) {
      m_item = iItem;
   }

private:
   FWGenParticleDetailView(const FWGenParticleDetailView&); // stop default
   const FWGenParticleDetailView& operator=(const FWGenParticleDetailView&); // stop default

   // ---------- member data --------------------------------
   const FWEventItem* m_item;
};


FWGenParticleDetailView::FWGenParticleDetailView ()
{
}

FWGenParticleDetailView::~FWGenParticleDetailView ()
{
}

void FWGenParticleDetailView::build (const FWModelId &id, const reco::GenParticle* iParticle, TEveWindowSlot* slot)
{
   if(0==iParticle) { return;}
 
   TEveScene* scene;
   TGLViewer* viewer;
   TGVerticalFrame* ediFrame;
   makePackViewer(slot, ediFrame, viewer, scene);

   m_item = id.item();
   // printf("calling ElectronDetailView::buildRhoZ\n");
   TEveTrackPropagator* rnrStyle = new TEveTrackPropagator();
   //units are Tesla
   rnrStyle->SetMagField( -4.0);
   //get this from geometry, units are CM
   rnrStyle->SetMaxR(120.0);
   rnrStyle->SetMaxZ(300.0);

   int index=0;
   //cout <<"----"<<endl;
   TEveRecTrack t;

   t.fBeta = 1.;
   t.fP = TEveVector(iParticle->px(),
                     iParticle->py(),
                     iParticle->pz());
   t.fV = TEveVector(iParticle->vx(),
                     iParticle->vy(),
                     iParticle->vz());
   t.fSign = iParticle->charge();

   TEveElementList* genPartList = new TEveElementList(Form("genParticle%d",index));
   scene->AddElement(genPartList);
   TEveTrack* genPart = new TEveTrack(&t,rnrStyle);
   genPart->SetMainColor(m_item->defaultDisplayProperties().color());
   genPart->MakeTrack();
   genPartList->AddElement(genPart);
}

REGISTER_FWDETAILVIEW(FWGenParticleDetailView);
