// ROOT includes
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLEmbeddedViewer.h"
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrans.h"
#include "TEveText.h"
#include "TEveGeoShape.h"
#include "TGLFontManager.h"
#include "TGPack.h"
#include "TGeoBBox.h"

// CMSSW includes
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// Fireworks includes
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/FWColorManager.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/FWGUISubviewArea.h"

#include "Fireworks/Tracks/plugins/FWTrackHitsDetailView.h"
#include "Fireworks/Tracks/plugins/TracksRecHitsUtil.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/CmsMagField.h"

FWTrackHitsDetailView::FWTrackHitsDetailView ():
m_viewer(0)
{
}

FWTrackHitsDetailView::~FWTrackHitsDetailView ()
{
    getEveWindow()->DestroyWindow();
}

void
FWTrackHitsDetailView::pickCameraCenter()
{
   m_viewer->PickCameraCenter();
}

void
FWTrackHitsDetailView::switchRenderStyle()
{
   if ( m_viewer->Style() == TGLRnrCtx::kWireFrame)
      m_viewer->SetStyle(TGLRnrCtx::kOutline);
   else
      m_viewer->SetStyle(TGLRnrCtx::kWireFrame);
   
   m_viewer->RequestDraw();
}

void
FWTrackHitsDetailView::build (const FWModelId &id, const reco::Track* track, TEveWindowSlot* base)
{   
   TEveScene*  scene(0);
   TEveViewer* viewer(0);
   TCanvas*    canvas(0);
   TGVerticalFrame* guiFrame;
   TEveWindow* eveWindow = FWDetailViewBase::makePackViewerGui(base, canvas, guiFrame, viewer, scene);
   eveWindow->SetElementName("Track Hits Detail View");
   setEveWindow(eveWindow);
   m_viewer = (TGLEmbeddedViewer*)viewer->GetGLViewer();
   
   {
      CSGAction* action = new CSGAction(this, "pickCameraCenter");
      action->createTextButton(guiFrame);
      action->activated.connect(sigc::mem_fun(this, &FWTrackHitsDetailView::pickCameraCenter));
   }
   {
      CSGAction* action = new CSGAction(this, "switchRenderStyle");
      action->createTextButton(guiFrame);
      action->activated.connect(sigc::mem_fun(this, &FWTrackHitsDetailView::switchRenderStyle));   
   }
   TGCompositeFrame* p = (TGCompositeFrame*)guiFrame->GetParent();
   p->MapSubwindows();
   p->Layout();
    
    TracksRecHitsUtil::addHits(*track, id.item(), scene);
   for (TEveElement::List_i i=scene->BeginChildren(); i!=scene->EndChildren(); ++i)
   {
      TEveGeoShape* gs = dynamic_cast<TEveGeoShape*>(*i);
      gs->SetMainColor(kBlue);
      gs->SetMainTransparency(0);
      gs->SetPickable(kFALSE);

      TEveText* text = new TEveText(gs->GetElementTitle());
      text->PtrMainTrans()->SetFrom(gs->RefMainTrans().Array());
      text->SetFontMode(TGLFont::kPolygon);

      TGeoBBox* bb = (TGeoBBox*)gs->GetShape();
      text->RefMainTrans().Move3LF(0, 0, 2*bb->GetDZ());
      text->PtrMainTrans()->RotateLF(2, 1, TMath::PiOver2());
      Double_t sx, sy, sz; text->PtrMainTrans()->GetScale(sx, sy, sz);
      Float_t a = 0.1*bb->GetDX()/text->GetFontSize();
      text->RefMainTrans().Scale(a, a, 1);
      gs->AddElement(text); 
   }

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
   trk->SetLineWidth(2);
   prop->SetRnrDaughters(kTRUE);
   prop->SetRnrDecay(kTRUE);
   prop->SetRnrReferences(kTRUE);
   prop->SetRnrDecay(kTRUE);
   prop->SetRnrFV(kTRUE);
   scene->AddElement(trk);

   // std::vector<TVector3> monoPoints;
   // std::vector<TVector3> stereoPoints;
   // fireworks::pushTrackerHits(monoPoints, stereoPoints, id, *track);
   // TEveElementList* list = new TEveElementList("hits");
   // fireworks::addTrackerHits3D(monoPoints, list, kRed, 1);
   // fireworks::addTrackerHits3D(stereoPoints, list, kRed, 1);
   // scene->AddElement(list);
   scene->Repaint(true);

   m_viewer->UpdateScene();
   m_viewer->CurrentCamera().Reset();
   m_viewer->RequestDraw(TGLRnrCtx::kLODHigh);
   m_viewer->SetStyle(TGLRnrCtx::kOutline);
   m_viewer->SetDrawCameraCenter(kTRUE);
}

void
FWTrackHitsDetailView::setBackgroundColor(Color_t col)
{
   FWColorManager::setColorSetViewer(m_viewer, col);
}

REGISTER_FWDETAILVIEW(FWTrackHitsDetailView, Hits);
