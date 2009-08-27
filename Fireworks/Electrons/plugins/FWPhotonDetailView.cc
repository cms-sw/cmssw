//
// Package:     Calo
// Class  :     FWPhotonDetailView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWPhotonDetailView.cc,v 1.11 2009/08/22 17:10:22 amraktad Exp $


#include "Fireworks/Electrons/plugins//FWPhotonDetailView.h"
#include "Fireworks/Electrons/plugins/FWECALDetailViewJohannes.icc"
#include "Fireworks/Electrons/plugins/FWECALDetailViewLothar.icc"
#include "Fireworks/Electrons/plugins/FWECALDetailViewDave.icc"

#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"

//
// constructors and destructor
//
FWPhotonDetailView::FWPhotonDetailView()
{
}

FWPhotonDetailView::~FWPhotonDetailView()
{
}

//
// member functions
//
void FWPhotonDetailView::build (const FWModelId &id, const reco::Photon* iPhoton, TEveWindowSlot* base)
{
   if(0==iPhoton) return;

   TEveWindowTab* eveWindow = base->MakeTab();
   eveWindow->SetShowTitleBar(kFALSE);

   TEveWindowSlot* slot;
   TEveWindow* ew;
   TEveScene*      scene;
   TEveViewer*     viewer;
   TGVerticalFrame* ediFrame;

   slot = eveWindow->NewSlot();
   ew = FWDetailViewBase::makePackViewer(slot, ediFrame, viewer, scene);
   ew->SetElementName("View A");
   FWECALDetailViewJohannes<reco::Photon>* viewJohannes =  new  FWECALDetailViewJohannes<reco::Photon>();
   viewJohannes->build(id, iPhoton, ediFrame, scene, viewer);

   slot = eveWindow->NewSlot();
   ew = FWDetailViewBase::makePackViewer(slot, ediFrame, viewer, scene);
   ew->SetElementName("View B");
   FWECALDetailViewLothar<reco::Photon>* viewLothar = new  FWECALDetailViewLothar<reco::Photon>();
   viewLothar->build(id, iPhoton, ediFrame, scene, viewer);

   // dave
   slot = eveWindow->NewSlot();
   ew = FWDetailViewBase::makePackViewer(slot, ediFrame, viewer, scene);
   ew->SetElementName("View C");
   FWECALDetailViewDave<reco::Photon>* viewDave = new  FWECALDetailViewDave<reco::Photon>();
   viewDave->build(id, iPhoton, ediFrame, scene, viewer);


   // eveWindow->GetTab()->SetTab(1);
}


REGISTER_FWDETAILVIEW(FWPhotonDetailView);
