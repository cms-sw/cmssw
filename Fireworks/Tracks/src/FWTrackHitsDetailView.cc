// ROOT includes
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TEveManager.h"
#include "TRootEmbeddedCanvas.h"
#include "TEveTrack.h"

// CMSSW includes
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// Fireworks includes
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDetailView.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Tracks/interface/FWTrackHitsDetailView.h"
#include "Fireworks/Tracks/plugins/TracksRecHitsUtil.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/CmsMagField.h"

FWTrackHitsDetailView::FWTrackHitsDetailView ()
{
}

FWTrackHitsDetailView::~FWTrackHitsDetailView ()
{
}

void
FWTrackHitsDetailView::build (const FWModelId &id, const reco::Track* track, TEveWindowSlot* slot)
{
   TEveViewer*  viewer = new TEveViewer("Track hits detail view");
   viewer->SpawnGLEmbeddedViewer();
   slot->ReplaceWindow(viewer);
   TEveScene* scene = gEve->SpawnNewScene("hits scene");
   viewer->AddScene(scene);
   viewer->SetShowTitleBar(kFALSE);

   TracksRecHitsUtil::addHits(*track, id.item(), scene);
   CmsMagField* cmsMagField = new CmsMagField;
   cmsMagField->setReverseState( true );
   cmsMagField->setMagnetState( CmsShowMain::getMagneticField() > 0 );

   TEveTrackPropagator* propagator = new TEveTrackPropagator;
   propagator->SetMagFieldObj( cmsMagField );
   propagator->SetStepper(TEveTrackPropagator::kRungeKutta);
   propagator->SetStepper(TEveTrackPropagator::kRungeKutta);
   propagator->SetMaxR(123);
   propagator->SetMaxZ(300);
   TEveTrack* trk = fireworks::prepareTrack( *track, propagator,
                                             id.item()->defaultDisplayProperties().color() );
   trk->MakeTrack();
   scene->AddElement(trk);

   /*
      std::vector<TVector3> monoPoints;
      std::vector<TVector3> stereoPoints;
      fireworks::pushTrackerHits(monoPoints, stereoPoints, id, *track);
      TEveElementList* list = new TEveElementList("hits");
      fireworks::addTrackerHits3D(monoPoints, list, id.item()->defaultDisplayProperties().color(), 1);
      fireworks::addTrackerHits3D(stereoPoints, list, id.item()->defaultDisplayProperties().color(), 2);
      scene->AddElement(list);
    */

   scene->Repaint(true);
   viewer->GetGLViewer()->UpdateScene();
   viewer->GetGLViewer()->CurrentCamera().Reset();
   viewer->GetGLViewer()->RequestDraw(TGLRnrCtx::kLODHigh);
   gEve->Redraw3D();
}
REGISTER_FWDETAILVIEW(FWTrackHitsDetailView);
