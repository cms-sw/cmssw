// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWTrackDetailView
// $Id: FWTrackDetailView.cc,v 1.19 2009/10/06 11:24:32 amraktad Exp $
//

#include "TEveLegoEventHandler.h"

// ROOT includes
#include "TLatex.h"
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TEveManager.h"
#include "TRootEmbeddedCanvas.h"

// CMSSW includes
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

// Fireworks includes
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/plugins/FWTrackDetailView.h"
#include "Fireworks/Tracks/interface/FWTrackResidualDetailView.h"
#include "Fireworks/Tracks/interface/FWTrackHitsDetailView.h"

//
// constructors and destructor
//
FWTrackDetailView::FWTrackDetailView():
m_hitsView(0)
{
}

FWTrackDetailView::~FWTrackDetailView()
{
   delete m_hitsView;
}

//
// member functions
//
void FWTrackDetailView::build(const FWModelId &id, const reco::Track* iTrack, TEveWindowSlot* base)
{
   if(0==iTrack) return;
   TEveWindowPack* eveWindow = base->MakePack();
   TEveCompositeFrame* eveFrame = eveWindow->GetEveFrame();
   TGMainFrame* parent  = (TGMainFrame*)eveFrame->GetParent();
   parent->Resize(790, 450);
   eveFrame->Layout();
   eveWindow->SetShowTitleBar(kFALSE);
   eveWindow->SetHorizontal();
   FWDetailViewBase::setEveWindow(eveWindow);

   TEveWindowSlot* slot;
   ////////////////////////////////////////////////////////////////////////
   //                              Sub-view 1
   ///////////////////////////////////////////////////////////////////////
  
   // prepare window
   slot = eveWindow->NewSlotWithWeight(40);
   FWTrackResidualDetailView builder1;
   builder1.build(id,iTrack,slot);
 
   ////////////////////////////////////////////////////////////////////////
   //                              Sub-view 2
   ///////////////////////////////////////////////////////////////////////
   // this view has to  be on heap, since it has to live to 
   // handle action signals
   slot = eveWindow->NewSlotWithWeight(60);
   m_hitsView = new FWTrackHitsDetailView();
   m_hitsView->build(id,iTrack,slot);
}

REGISTER_FWDETAILVIEW(FWTrackDetailView);
