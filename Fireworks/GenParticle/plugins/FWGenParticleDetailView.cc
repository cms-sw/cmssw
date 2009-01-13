#include "TEveElement.h"
#include "TEveGeoNode.h"
#include "TEveManager.h"
#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"
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
   
   virtual TEveElement* build (const FWModelId &id, const reco::GenParticle*);
   
protected:
   void setItem (const FWEventItem *iItem) { m_item = iItem; }
   
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

TEveElement* FWGenParticleDetailView::build (const FWModelId &id, const reco::GenParticle* iParticle)
{
   if(0==iParticle) { return 0;}
   m_item = id.item();
   // printf("calling ElectronDetailView::buildRhoZ\n");
   TEveElementList* tList =   new TEveElementList(m_item->name().c_str(),"Supercluster RhoZ",true);
   tList->SetMainColor(m_item->defaultDisplayProperties().color());
   
   TEveTrackPropagator* rnrStyle = new TEveTrackPropagator();
   //units are Tesla
   rnrStyle->SetMagField( -4.0);
   //get this from geometry, units are CM
   rnrStyle->SetMaxR(120.0);
   rnrStyle->SetMaxZ(300.0);
   
   //  Original Commented out here
   //  TEveTrackPropagator* rnrStyle = tList->GetPropagator();
   
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
   gEve->AddElement(genPartList,tList);
   TEveTrack* genPart = new TEveTrack(&t,rnrStyle);
   genPart->SetMainColor(m_item->defaultDisplayProperties().color());
   genPart->MakeTrack();
   genPartList->AddElement(genPart);
   
   return tList;
}

REGISTER_FWDETAILVIEW(FWGenParticleDetailView);
