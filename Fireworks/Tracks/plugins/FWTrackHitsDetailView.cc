// ROOT includes
#include "TGLFontManager.h"
#include "TEveScene.h"
#include "TEveManager.h"
#include "TEveStraightLineSet.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveTrans.h"
#include "TEveText.h"
#include "TEveGeoShape.h"
#include "TGSlider.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGLViewer.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TLegend.h"

// CMSSW includes
#include "DataFormats/TrackReco/interface/Track.h"

// Fireworks includes
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWColorManager.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/FWIntValueListener.h"
#include "Fireworks/Core/interface/FWMagField.h"

#include "Fireworks/Tracks/plugins/FWTrackHitsDetailView.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

FWTrackHitsDetailView::FWTrackHitsDetailView ():
  m_modules(0),
  m_moduleLabels(0),
  m_hits(0),
  m_slider(0),
  m_sliderListener(),
  m_legend(0)
{}

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
   makeLegend();
   
   TGCompositeFrame* p = (TGCompositeFrame*)m_guiFrame->GetParent();
   p->MapSubwindows();
   p->Layout();

   m_modules = new TEveElementList("Modules");
   m_eveScene->AddElement(m_modules);
   m_moduleLabels = new TEveElementList("Modules");
   m_eveScene->AddElement(m_moduleLabels);
   if( track->extra().isAvailable() )
      fireworks::addModules(*track, id.item(), m_modules, true);
   m_hits = new TEveElementList("Hits");
   m_eveScene->AddElement(m_hits);
   if( track->extra().isAvailable() )
      fireworks::addHits(*track, id.item(), m_hits, true);
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
   prop->SetMagFieldObj(item()->context().getField(), false);
   prop->SetStepper(TEveTrackPropagator::kRungeKutta);
   prop->SetMaxR(123);
   prop->SetMaxZ(300);
   prop->SetMaxStep(1);
   TEveTrack* trk = fireworks::prepareTrack( *track, prop );
   trk->MakeTrack();
   trk->SetLineWidth(2);
   trk->SetTitle( "Track and its ref states" );
   prop->SetRnrDaughters(kTRUE);
   prop->SetRnrReferences(kTRUE);
   prop->SetRnrDecay(kTRUE);
   prop->SetRnrFV(kTRUE);
   trk->SetMainColor(id.item()->defaultDisplayProperties().color());
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

   // m_eveScene->Repaint(true);

   viewerGL()->SetStyle(TGLRnrCtx::kOutline);
   viewerGL()->SetDrawCameraCenter(kTRUE);
   viewerGL()->ResetCamerasAfterNextUpdate();
   viewerGL()->UpdateScene(kFALSE);
   gEve->Redraw3D();

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
FWTrackHitsDetailView::setTextInfo(const FWModelId &id, const reco::Track* track)
{
   m_infoCanvas->cd();

   float_t x  = 0.02;
   float   y  = 0.95;

   TLatex* latex = new TLatex( x, y, "" );
   const double textsize( 0.07 );
   latex->SetTextSize( 2*textsize );

   latex->DrawLatex( x, y, id.item()->modelName( id.index()).c_str());
   y -= latex->GetTextSize()*0.6;

   latex->SetTextSize( textsize );
   float lineH = latex->GetTextSize()*0.6;

   latex->DrawLatex( x, y, Form( " P_{T} = %.1f GeV, #eta = %0.2f, #varphi = %0.2f",
				 track->pt(), track->eta(), track->phi()));
   y -= lineH;

   if( track->charge() > 0 )
      latex->DrawLatex( x, y, " charge = +1" );
   else
      latex->DrawLatex( x, y, " charge = -1" );
   y -= lineH;
   y -= lineH;

   latex->DrawLatex( x, y, "Track modules:");
   y -= lineH;

   Double_t pos[4];
   pos[0] = x+0.05;
   pos[2] = x+0.20;
   Double_t boxH = 0.25*textsize;

   pos[1] = y; pos[3] = pos[1] + boxH;
   FWDetailViewBase::drawCanvasBox( pos, kBlue );
   latex->DrawLatex( x + 0.25, y, "Module" );
   y -= lineH;

   pos[1] = y; pos[3] = pos[1] + boxH;
   FWDetailViewBase::drawCanvasBox( pos, kRed );
   latex->DrawLatex( x + 0.25, y, "LOST Module" );
   y -= lineH;

   pos[1] = y; pos[3] = pos[1] + boxH;
   FWDetailViewBase::drawCanvasBox( pos, 28 );
   latex->DrawLatex( x + 0.25, y, "INACTIVE Module" );
   y -= lineH;

   pos[1] = y; pos[3] = pos[1] + boxH;
   FWDetailViewBase::drawCanvasBox( pos, 218 );
   latex->DrawLatex( x + 0.25, y, "BAD Module" );
   y -= lineH;

   Float_t r = 0.01;
   Float_t r2 = 0.02;
   y -= lineH;
   drawCanvasDot( x + r2, y, r2, kGreen );
   y -= r;
   latex->DrawLatex( x + 3 * r2, y, "Pixel Hits" );
   y -= lineH;

   drawCanvasDot( x + r2, y, r2, kRed);
   y -= r;
   latex->DrawLatex( x + 3 * r2, y, "Extra Pixel Hits" );
   y -= lineH;

   m_legend->SetY2(y);
   m_legend->Draw();
   m_legend = 0; // Deleted together with TPad.
}

void
FWTrackHitsDetailView::makeLegend( void )
{
   m_legend = new TLegend( 0.01, 0.01, 0.99, 0.99, 0, "NDC" );
   m_legend->SetTextSize( 0.07 );
   m_legend->SetBorderSize( 0 );
   m_legend->SetMargin( 0.15 );
   m_legend->SetEntrySeparation( 0.01 );

   TEveStraightLineSet *legend = new TEveStraightLineSet( "siStripCluster" );
   legend->SetLineWidth( 3 );
   legend->SetLineColor( kGreen );
   m_legend->AddEntry( legend, "Exact SiStripCluster", "l");

   TEveStraightLineSet *legend2 = new TEveStraightLineSet( "siStripCluster2" );
   legend2->SetLineWidth( 3 );
   legend2->SetLineColor( kRed );
   m_legend->AddEntry( legend2, "Extra SiStripCluster", "l");

   TEveStraightLineSet *legend3 = new TEveStraightLineSet( "refStates" );
   legend3->SetDepthTest( kFALSE );
   legend3->SetMarkerColor( kYellow );
   legend3->SetMarkerStyle( kPlus );
   legend3->SetMarkerSize( 2 );
   m_legend->AddEntry( legend3, "Inner/Outermost States", "p");

   TEveStraightLineSet *legend4 = new TEveStraightLineSet( "vertex" );
   legend4->SetDepthTest( kFALSE );
   legend4->SetMarkerColor( kRed );
   legend4->SetMarkerStyle( kFullDotLarge );
   legend4->SetMarkerSize( 2 );
   m_legend->AddEntry( legend4, "Vertex", "p");

   TEveStraightLineSet *legend5 = new TEveStraightLineSet( "cameraCenter" );
   legend5->SetDepthTest( kFALSE );
   legend5->SetMarkerColor( kCyan );
   legend5->SetMarkerStyle( kFullDotLarge );
   legend5->SetMarkerSize( 2 );
   m_legend->AddEntry( legend5, "Camera center", "p");
}   

void
FWTrackHitsDetailView::rnrLabels()
{
   m_moduleLabels->SetRnrChildren(!m_moduleLabels->GetRnrChildren());
   gEve->Redraw3D();
}

REGISTER_FWDETAILVIEW(FWTrackHitsDetailView, Hits);
