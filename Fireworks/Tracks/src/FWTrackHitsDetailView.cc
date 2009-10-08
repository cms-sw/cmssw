// ROOT includes
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLEmbeddedViewer.h"
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TGPack.h"

// CMSSW includes
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// Fireworks includes
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/src/CmsShowMain.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/FWGUISubviewArea.h"

#include "Fireworks/Tracks/interface/FWTrackHitsDetailView.h"
#include "Fireworks/Tracks/plugins/TracksRecHitsUtil.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/CmsMagField.h"

FWTrackHitsDetailView::FWTrackHitsDetailView ():
m_viewer(0)
{
}

FWTrackHitsDetailView::~FWTrackHitsDetailView ()
{
}

void
FWTrackHitsDetailView::pickCameraCenter()
{
   m_viewer->PickCameraCenter();
}

void
FWTrackHitsDetailView::build (const FWModelId &id, const reco::Track* track, TEveWindowSlot* base)
{
   TEveViewer*  ev = new TEveViewer("Track hits detail view");
   m_viewer = ev->SpawnGLEmbeddedViewer();
   base->ReplaceWindow(ev);
   TEveScene* scene = gEve->SpawnNewScene("hits scene");
   ev->AddScene(scene);
   setEveWindow(ev);

   FWGUISubviewArea* toolBar = FWGUISubviewArea::getToolBarFromWindow(ev);
   CSGAction* action = new CSGAction(this, "pickCameraCenter");
   // layout hints kLeft does not work, have to do more complicated way
   TGTextButton* textButton = new TGTextButton(toolBar, "pickCameraCenter");
   TList* flist = toolBar->GetList();
   TGFrameElement *nw = new TGFrameElement(textButton, new TGLayoutHints( kLHintsNormal));
   flist->AddFirst(nw);
   toolBar->MapSubwindows();
   toolBar->Layout();

   TQObject::Connect(textButton, "Clicked()", "CSGAction", action, "activate()");
   action->activated.connect(sigc::mem_fun(this, &FWTrackHitsDetailView::pickCameraCenter));

   TracksRecHitsUtil::addHits(*track, id.item(), scene);
   CmsMagField* cmsMagField = new CmsMagField;
   cmsMagField->setReverseState( true );
   cmsMagField->setMagnetState( CmsShowMain::getMagneticField() > 0 );

   TEveTrackPropagator* prop = new TEveTrackPropagator();
   prop->SetMagFieldObj( cmsMagField );
   prop->SetStepper(TEveTrackPropagator::kRungeKutta);
   prop->SetMaxR(123);
   prop->SetMaxZ(300);
   TEveTrack* trk = fireworks::prepareTrack( *track, prop,
                                             id.item()->defaultDisplayProperties().color() );
   trk->MakeTrack();
   prop->SetRnrDaughters(kTRUE);
   prop->SetRnrDecay(kTRUE);
   prop->SetRnrReferences(kTRUE);
   prop->SetRnrDecay(kTRUE);
   prop->SetRnrFV(kTRUE);
   scene->AddElement(trk);

   std::vector<TVector3> monoPoints;
   std::vector<TVector3> stereoPoints;
   fireworks::pushTrackerHits(monoPoints, stereoPoints, id, *track);
   TEveElementList* list = new TEveElementList("hits");
   fireworks::addTrackerHits3D(monoPoints, list, kRed, 1);
   fireworks::addTrackerHits3D(stereoPoints, list, kRed, 1);
   scene->AddElement(list);
   scene->Repaint(true);

   m_viewer->UpdateScene();
   m_viewer->CurrentCamera().Reset();
   m_viewer->RequestDraw(TGLRnrCtx::kLODHigh);
   m_viewer->SetStyle(TGLRnrCtx::kOutline);
   m_viewer->SetDrawCameraCenter(kTRUE);
}

