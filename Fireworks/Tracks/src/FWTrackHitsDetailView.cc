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
   printf("Pick camera center.\n");fflush(stdout);
   m_viewer->PickCameraCenter();
}

void
FWTrackHitsDetailView::build (const FWModelId &id, const reco::Track* track, TEveWindowSlot* base)
{
   TEveWindowPack* wp = base->MakePack();
   wp->SetShowTitleBar(kFALSE);
   TGPack* pack = wp->GetPack();
   pack->SetUseSplitters(kFALSE);

   // GUI
   TGCompositeFrame* guiFrame = new TGVerticalFrame(pack);
   pack->AddFrameWithWeight(guiFrame, new TGLayoutHints(kLHintsNormal),0.2);

   CSGAction* actionRnr = new CSGAction(this, "pickCameraCenter");
   actionRnr->createTextButton(guiFrame, new TGLayoutHints(kLHintsNormal, 2, 2, 0, 0));
   actionRnr->activated.connect( sigc::mem_fun(this, &FWTrackHitsDetailView::pickCameraCenter));

   // view
   m_viewer = new TGLEmbeddedViewer(pack, 0, 0);
   TEveViewer* eveViewer= new TEveViewer("DetailViewViewer");
   eveViewer->SetGLViewer(m_viewer, m_viewer->GetFrame());
   pack->AddFrameWithWeight(m_viewer->GetFrame(),0, 5);
   TEveScene* scene = gEve->SpawnNewScene("Detailed view");
   eveViewer->AddScene(scene);

   pack->MapSubwindows();
   pack->Layout();
   pack->MapWindow();

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
