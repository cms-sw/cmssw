// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWTrackDetailView
// $Id: FWTrackDetailView.cc,v 1.35 2009/09/04 18:23:48 amraktad Exp $
//

#include "TEveLegoEventHandler.h"

// ROOT includes
#include "TLatex.h"
#include "TEveCalo.h"
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TEveManager.h"
#include "TCanvas.h" 
#include "TEveCaloLegoOverlay.h"
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
FWTrackDetailView::FWTrackDetailView()
{
}

FWTrackDetailView::~FWTrackDetailView()
{
}

//
// member functions
//
void FWTrackDetailView::build(const FWModelId &id, const reco::Track* iTrack, TEveWindowSlot* base)
{
   if(0==iTrack) return;
   TEveWindowPack* eveWindow = base->MakePack();
   eveWindow->SetShowTitleBar(kFALSE);

   // TEveWindow* ew(0);
   TEveWindowSlot* slot(0);
   // TEveScene*      scene(0);
   // TEveViewer*     viewer(0);
   // TGVerticalFrame* ediFrame(0);

   ////////////////////////////////////////////////////////////////////////
   //                              Sub-view 1
   ///////////////////////////////////////////////////////////////////////

   // prepare window
   slot = eveWindow->NewSlot();
   FWTrackResidualDetailView builder1;
   builder1.build(id,iTrack,slot);
   
   ////////////////////////////////////////////////////////////////////////
   //                              Sub-view 2
   ///////////////////////////////////////////////////////////////////////
   slot = eveWindow->NewSlot();
   FWTrackHitsDetailView builder2;
   builder2.build(id,iTrack,slot);
   
}

REGISTER_FWDETAILVIEW(FWTrackDetailView);
