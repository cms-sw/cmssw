// ROOT includes
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLEmbeddedViewer.h"
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveText.h"
#include "TEveGeoShape.h"
#include "TGLFontManager.h"
#include "TGPack.h"
#include "TGeoBBox.h"
#include "TGSlider.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TCanvas.h"
#include "TLatex.h"

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
#include "Fireworks/Core/interface/FWIntValueListener.h"
#include "Fireworks/Core/interface/FWMagField.h"

#include "Fireworks/Tracks/plugins/FWTrackHitsDetailView.h"
#include "Fireworks/Tracks/plugins/TracksRecHitsUtil.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

FWTrackHitsDetailView::FWTrackHitsDetailView ():
m_modules(0),
m_moduleLabels(0),
m_hits(0),
m_slider(0),
m_sliderListener()
{
}

FWTrackHitsDetailView::~FWTrackHitsDetailView ()
{
}

void
FWTrackHitsDetailView::build (const FWModelId &id, const reco::Track* track)
{      
   bool labelsOn = false;
   {
      TGCompositeFrame* f  = new TGVerticalFrame(m_guiFrame);
      m_guiFrame->AddFrame(f);
      f->AddFrame(new TGLabel(f, "Module Transparency:"), new TGLayoutHints(kLHintsLeft, 2, 2, 0, 0));
      m_slider = new TGHSlider(f, 120, kSlider1 | kScaleNo);
      f->AddFrame(m_slider, new TGLayoutHints(kLHintsTop | kLHintsLeft, 2, 2, 1, 4));
      m_slider->SetRange(0, 100);
      m_slider->SetPosition(75);

      m_sliderListener =  new FWIntValueListener();
      TQObject::Connect(m_slider, "PositionChanged(Int_t)", "FWIntValueListenerBase",  m_sliderListener, "setValue(Int_t)");
      m_sliderListener->valueChanged_.connect(boost::bind(&FWTrackHitsDetailView::transparencyChanged,this,_1));
   }

   {
      CSGAction* action = new CSGAction(this, "Show Module Labels");
      TGCheckButton* b = new TGCheckButton(m_guiFrame, "Show Module Labels" );
      b->SetState(labelsOn ? kButtonDown : kButtonUp, false);
      m_guiFrame->AddFrame(b, new TGLayoutHints( kLHintsNormal, 2, 3, 1, 4));
      TQObject::Connect(b, "Clicked()", "CSGAction", action, "activate()");
      action->activated.connect(sigc::mem_fun(this, &FWTrackHitsDetailView::rnrLabels));
   }
   {
      CSGAction* action = new CSGAction(this, " Pick Camera Center ");
      action->createTextButton(m_guiFrame, new TGLayoutHints( kLHintsNormal, 2, 0, 1, 4));
      action->setToolTip("Click on object in viewer to set camera center.");
      action->activated.connect(sigc::mem_fun(this, &FWTrackHitsDetailView::pickCameraCenter));
   }

   TGCompositeFrame* p = (TGCompositeFrame*)m_guiFrame->GetParent();
   p->MapSubwindows();
   p->Layout();
    
   m_modules = new TEveElementList("Modules");
   m_eveScene->AddElement(m_modules);
   m_moduleLabels = new TEveElementList("Modules");
   m_eveScene->AddElement(m_moduleLabels);
   TracksRecHitsUtil::addModules(*track, id.item(), m_modules, true);
   m_hits = new TEveElementList("Hits");
   m_eveScene->AddElement(m_hits);
   TracksRecHitsUtil::addHits(*track, id.item(), m_hits, true);
   for (TEveElement::List_i i=m_modules->BeginChildren(); i!=m_modules->EndChildren(); ++i)
   {
      TEveGeoShape* gs = dynamic_cast<TEveGeoShape*>(*i);
      if (gs == 0 && (*i != 0)) {
        std::cerr << "Got a " << typeid(**i).name() << ", expecting TEveGeoShape. ignoring (it must be the clusters)." << std::endl;
        continue;
      }
      gs->SetMainTransparency(75);
      gs->SetPickable(kFALSE);

      TString name = gs->GetElementTitle();
      if (!name.Contains("BAD") && !name.Contains("INACTIVE") && !name.Contains("LOST")) {
          gs->SetMainColor(kBlue);
      }
      TEveText* text = new TEveText(name.Data());
      text->PtrMainTrans()->SetFrom(gs->RefMainTrans().Array());
      text->SetFontMode(TGLFont::kPixmap);
      text->SetFontSize(12);
      m_moduleLabels->AddElement(text); 
   }
   m_moduleLabels->SetRnrChildren(labelsOn);

   TEveTrackPropagator* prop = new TEveTrackPropagator();
   prop->SetMagFieldObj (item()->context().getField());
   prop->SetStepper(TEveTrackPropagator::kRungeKutta);
   prop->SetMaxR(123);
   prop->SetMaxZ(300);
   prop->SetMaxStep(1);
   TEveTrack* trk = fireworks::prepareTrack( *track, prop,
                                             id.item()->defaultDisplayProperties().color() );
   trk->MakeTrack();
   trk->SetLineWidth(2);
   prop->SetRnrDaughters(kTRUE);
   prop->SetRnrDecay(kTRUE);
   prop->SetRnrReferences(kTRUE);
   prop->SetRnrDecay(kTRUE);
   prop->SetRnrFV(kTRUE);
   m_eveScene->AddElement(trk);


// -- add PixelHits
//LatB
//    std::vector<TVector3> pixelPoints;
//    fireworks::pushPixelHits(pixelPoints, *id.item(), *track);
//    TEveElementList* list = new TEveElementList("PixelHits");
//    fireworks::addTrackerHits3D(pixelPoints, list, kRed, 2);
//    m_eveScene->AddElement(list);
	
//    list = new TEveElementList("SiStripClusterHits");
// 	fireworks::addSiStripClusters(id.item(), *track, list, kRed);
//    m_eveScene->AddElement(list);
//LatB

   m_eveScene->Repaint(true);

   viewerGL()->UpdateScene();
   viewerGL()->CurrentCamera().Reset();
   viewerGL()->RequestDraw(TGLRnrCtx::kLODHigh);
   viewerGL()->SetStyle(TGLRnrCtx::kOutline);
   viewerGL()->SetDrawCameraCenter(kTRUE);

   setTextInfo(id, track);
}

void
FWTrackHitsDetailView::setBackgroundColor(Color_t col)
{
   // Callback for cmsShow change of background
   return;
   FWColorManager::setColorSetViewer(viewerGL(), col);

   // adopt label colors to background, this should be implemneted in TEveText
   if (m_moduleLabels)
   {
      Color_t x = viewerGL()->GetRnrCtx()->ColorSet().Foreground().GetColorIndex();
      for (TEveElement::List_i i=m_moduleLabels->BeginChildren(); i!=m_moduleLabels->EndChildren(); ++i)
         (*i)->SetMainColor(x);
   }
}


void
FWTrackHitsDetailView::pickCameraCenter()
{
   viewerGL()->PickCameraCenter();
}

void
FWTrackHitsDetailView::transparencyChanged(int x)
{
   for (TEveElement::List_i i=m_modules->BeginChildren(); i!=m_modules->EndChildren(); ++i)
   {
      (*i)->SetMainTransparency(x);
   }
   gEve->Redraw3D();
}

void
FWTrackHitsDetailView::setTextInfo(const FWModelId &id, const reco::Track*)
{
   m_infoCanvas->cd();

   Double_t fontSize = 0.07;
   TLatex* latex = new TLatex();
   latex->SetTextSize(fontSize);
   Double_t y = 0.9;
   Double_t x = 0.02;
   Double_t yStep = 0.04;
   latex->DrawLatex(x, y, "Track hits info:");
   y -= yStep;

   Float_t r = 0.02;
   drawCanvasDot(x + r, y, r, kCyan);
   y -= r*0.5;
   latex->DrawLatex(x+ 3*r, y, "Camera center");
}

void
FWTrackHitsDetailView::rnrLabels()
{
   m_moduleLabels->SetRnrChildren(!m_moduleLabels->GetRnrChildren());
   gEve->Redraw3D();
}

REGISTER_FWDETAILVIEW(FWTrackHitsDetailView, Hits);
