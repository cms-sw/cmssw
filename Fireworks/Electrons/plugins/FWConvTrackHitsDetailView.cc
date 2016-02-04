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
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

// Fireworks includes
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/FWIntValueListener.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/Electrons/plugins/FWConvTrackHitsDetailView.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

FWConvTrackHitsDetailView::FWConvTrackHitsDetailView ():
  m_modules(0),
  m_moduleLabels(0),
  m_hits(0),
  m_slider(0),
  m_sliderListener(),
  m_legend(0)
{}

FWConvTrackHitsDetailView::~FWConvTrackHitsDetailView ()
{
}

void
FWConvTrackHitsDetailView::build (const FWModelId &id, const reco::Conversion* conv)
{      
  if (conv->nTracks()<2) return;
  const reco::Track* track0 = conv->tracks().at(0).get();
  const reco::Track* track1 = conv->tracks().at(1).get();

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
      m_sliderListener->valueChanged_.connect(boost::bind(&FWConvTrackHitsDetailView::transparencyChanged,this,_1));
   }

   {
      CSGAction* action = new CSGAction(this, "Show Module Labels");
      TGCheckButton* b = new TGCheckButton(m_guiFrame, "Show Module Labels" );
      b->SetState(kButtonUp, false);
      m_guiFrame->AddFrame(b, new TGLayoutHints( kLHintsNormal, 2, 3, 1, 4));
      TQObject::Connect(b, "Clicked()", "CSGAction", action, "activate()");
      action->activated.connect(sigc::mem_fun(this, &FWConvTrackHitsDetailView::rnrLabels));
   }
   {
      CSGAction* action = new CSGAction(this, " Pick Camera Center ");
      action->createTextButton(m_guiFrame, new TGLayoutHints( kLHintsNormal, 2, 0, 1, 4));
      action->setToolTip("Click on object in viewer to set camera center.");
      action->activated.connect(sigc::mem_fun(this, &FWConvTrackHitsDetailView::pickCameraCenter));
   }
   //makeLegend(); //it seems there is not much room for this
   
   TGCompositeFrame* p = (TGCompositeFrame*)m_guiFrame->GetParent();
   p->MapSubwindows();
   p->Layout();

   m_modules = new TEveElementList( "Modules" );
   m_eveScene->AddElement( m_modules );
   m_moduleLabels = new TEveElementList( "Modules" );
   m_eveScene->AddElement( m_moduleLabels );
   m_hits = new TEveElementList( "Hits" );
   m_eveScene->AddElement( m_hits );
   if( track1->extra().isAvailable())
   {
      addModules( *track1, id.item(), m_modules, true );
      addHits( *track1, id.item(), m_hits, true );
   }
   if( track0->extra().isAvailable())
   {
      addModules( *track0, id.item(), m_modules, true );
      addHits( *track0, id.item(), m_hits, true );
   }
   for( TEveElement::List_i i = m_modules->BeginChildren(), end = m_modules->EndChildren(); i != end; ++i )
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
   m_moduleLabels->SetRnrChildren(false);

   TEveTrackPropagator* prop = new TEveTrackPropagator();
   prop->SetMagFieldObj(item()->context().getField(), false);
   prop->SetStepper(TEveTrackPropagator::kRungeKutta);
   prop->SetMaxR(123);
   prop->SetMaxZ(300);
   prop->SetMaxStep(1);
   prop->SetRnrDaughters(kTRUE);
   prop->SetRnrReferences(kTRUE);
   prop->SetRnrDecay(kTRUE);
   prop->SetRnrFV(kTRUE);

   TEveTrack* trk1 = fireworks::prepareTrack( *track1, prop );
   trk1->MakeTrack();
   trk1->SetLineWidth(2);
   trk1->SetTitle( "Track 1 and its ref states" );
   trk1->SetMainColor(id.item()->defaultDisplayProperties().color());
   m_eveScene->AddElement(trk1);

   TEveTrack* trk0 = fireworks::prepareTrack( *track0, prop );
   trk0->MakeTrack();
   trk0->SetLineWidth(2);
   trk0->SetTitle( "Track 0 and its ref states" );
   trk0->SetMainColor(id.item()->defaultDisplayProperties().color());
   m_eveScene->AddElement(trk0);

   viewerGL()->SetStyle(TGLRnrCtx::kOutline);
   viewerGL()->SetDrawCameraCenter(kTRUE);
   viewerGL()->ResetCamerasAfterNextUpdate();
   viewerGL()->UpdateScene(kFALSE);
   gEve->Redraw3D();

   setTextInfo(id, conv);
}

void
FWConvTrackHitsDetailView::setBackgroundColor(Color_t col)
{
   // Callback for cmsShow change of background

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
FWConvTrackHitsDetailView::pickCameraCenter()
{
   viewerGL()->PickCameraCenter();
}

void
FWConvTrackHitsDetailView::transparencyChanged(int x)
{
   for (TEveElement::List_i i=m_modules->BeginChildren(); i!=m_modules->EndChildren(); ++i)
   {
      (*i)->SetMainTransparency(x);
   }
   gEve->Redraw3D();
}

void
FWConvTrackHitsDetailView::setTextInfo(const FWModelId &id, const reco::Conversion* conv)
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

   latex->DrawLatex( x, y, Form("p_{T}=%.1f GeV, #eta=%0.2f, #varphi=%0.2f",
				sqrt(conv->pairMomentum().Perp2()), conv->pairMomentum().eta(), conv->pairMomentum().phi() ));
   y -= lineH;
   latex->DrawLatex( x, y, Form("vtx=(%.1f, %.1f, %.1f) r=%.1f [cm]",
				conv->conversionVertex().x(), conv->conversionVertex().y(), conv->conversionVertex().z(), 
				conv->conversionVertex().position().rho() ));
   y -= lineH;
   latex->DrawLatex( x, y, Form("#Deltactg#theta=%.3f",
				conv->pairCotThetaSeparation() ));
   y -= lineH;
   latex->DrawLatex( x, y, Form("#Delta#phi_{vtx}=%.3f",
				conv->dPhiTracksAtVtx() ));
   y -= lineH;
   latex->DrawLatex( x, y, Form("dist. min. app.=%.3f cm",
				conv->distOfMinimumApproach() ));
   y -= lineH;
   y -= lineH;


   const reco::Track* track0 = conv->tracks().at(0).get();
   latex->DrawLatex( x, y, Form("Trk0 q=%i",track0->charge()));
   y -= lineH;
   latex->DrawLatex( x, y, Form( "p_{T}=%.1f GeV, #eta=%0.2f, #varphi=%0.2f",
				 track0->pt(), track0->eta(), track0->phi()));
   y -= lineH;
   const reco::HitPattern& p_tk0 = track0->hitPattern();
   int nvalid_tk0=0, ninvalid_tk0=0, npix_tk0=0, nstrip_tk0=0;
   for(int i_tk0=0; i_tk0<p_tk0.numberOfHits(); i_tk0++) {
     uint32_t hit = p_tk0.getHitPattern(i_tk0);
     if(p_tk0.validHitFilter(hit)) {
       nvalid_tk0++;
       if (p_tk0.pixelHitFilter(hit)) npix_tk0++;
       else if (p_tk0.stripHitFilter(hit)) nstrip_tk0++;
     } else ninvalid_tk0++;
   }
   latex->DrawLatex( x, y,  Form( "valid hits: %i (pix. %i, str. %i)", nvalid_tk0, npix_tk0, nstrip_tk0) );
   y -= lineH;
   latex->DrawLatex( x, y,  Form( "invalid: %i", ninvalid_tk0) );
   y -= lineH;
   const reco::HitPattern& p_mhi_tk0 = track0->trackerExpectedHitsInner();
   int npix_mhi_tk0=0, nstrip_mhi_tk0=0;
   for(int i_mhi_tk0=0; i_mhi_tk0<p_mhi_tk0.numberOfHits(); i_mhi_tk0++) {
     uint32_t hit = p_mhi_tk0.getHitPattern(i_mhi_tk0);
     if (p_mhi_tk0.pixelHitFilter(hit)) npix_mhi_tk0++;
     else if (p_mhi_tk0.stripHitFilter(hit)) nstrip_mhi_tk0++;
   }
   latex->DrawLatex( x, y,  Form("miss. inner hits: pix. %i, str. %i", npix_mhi_tk0, nstrip_mhi_tk0) );

   y -= lineH;
   y -= lineH;
   const reco::Track* track1 = conv->tracks().at(1).get();
   latex->DrawLatex( x, y, Form("Trk1 q=%i",track1->charge()));
   y -= lineH;
   latex->DrawLatex( x, y, Form( "p_{T}=%.1f GeV, #eta=%0.2f, #varphi=%0.2f",
				 track1->pt(), track1->eta(), track1->phi()));
   y -= lineH;
   const reco::HitPattern& p_tk1 = track1->hitPattern();
   int nvalid_tk1=0, ninvalid_tk1=0, npix_tk1=0, nstrip_tk1=0;
   for(int i_tk1=0; i_tk1<p_tk1.numberOfHits(); i_tk1++) {
     uint32_t hit = p_tk1.getHitPattern(i_tk1);
     if(p_tk1.validHitFilter(hit)) {
       nvalid_tk1++;
       if (p_tk1.pixelHitFilter(hit)) npix_tk1++;
       else if (p_tk1.stripHitFilter(hit)) nstrip_tk1++;
     } else ninvalid_tk1++;
   }
   latex->DrawLatex( x, y,  Form( "valid hits: %i (pix. %i, str. %i)", nvalid_tk1, npix_tk1, nstrip_tk1) );
   y -= lineH;
   latex->DrawLatex( x, y,  Form( "invalid: %i", ninvalid_tk1) );
   y -= lineH;
   const reco::HitPattern& p_mhi_tk1 = track1->trackerExpectedHitsInner();
   int npix_mhi_tk1=0, nstrip_mhi_tk1=0;
   for(int i_mhi_tk1=0; i_mhi_tk1<p_mhi_tk1.numberOfHits(); i_mhi_tk1++) {
     uint32_t hit = p_mhi_tk1.getHitPattern(i_mhi_tk1);
     if (p_mhi_tk1.pixelHitFilter(hit)) npix_mhi_tk1++;
     else if (p_mhi_tk1.stripHitFilter(hit)) nstrip_mhi_tk1++;
   }
   latex->DrawLatex( x, y,  Form("miss. inner hits: pix. %i, str. %i", npix_mhi_tk1, nstrip_mhi_tk1) );
   y -= lineH;
   y -= lineH;

   latex->DrawLatex( x, y, "Placeholder for symbol legend");
   y -= lineH;
   latex->DrawLatex( x, y, "and projection buttons");
//    latex->DrawLatex( x, y, "Track modules:");
//    y -= lineH;

//    Double_t pos[4];
//    pos[0] = x+0.05;
//    pos[2] = x+0.20;
//    Double_t boxH = 0.25*textsize;

//    pos[1] = y; pos[3] = pos[1] + boxH;
//    FWDetailViewBase::drawCanvasBox( pos, kBlue );
//    latex->DrawLatex( x + 0.25, y, "Module" );
//    y -= lineH;

//    pos[1] = y; pos[3] = pos[1] + boxH;
//    FWDetailViewBase::drawCanvasBox( pos, kRed );
//    latex->DrawLatex( x + 0.25, y, "LOST Module" );
//    y -= lineH;

//    pos[1] = y; pos[3] = pos[1] + boxH;
//    FWDetailViewBase::drawCanvasBox( pos, 28 );
//    latex->DrawLatex( x + 0.25, y, "INACTIVE Module" );
//    y -= lineH;

//    pos[1] = y; pos[3] = pos[1] + boxH;
//    FWDetailViewBase::drawCanvasBox( pos, 218 );
//    latex->DrawLatex( x + 0.25, y, "BAD Module" );
//    y -= lineH;

//    Float_t r = 0.01;
//    Float_t r2 = 0.02;
//    y -= lineH;
//    drawCanvasDot( x + r2, y, r2, kGreen );
//    y -= r;
//    latex->DrawLatex( x + 3 * r2, y, "Pixel Hits" );
//    y -= lineH;

//    drawCanvasDot( x + r2, y, r2, kRed);
//    y -= r;
//    latex->DrawLatex( x + 3 * r2, y, "Extra Pixel Hits" );
//    y -= lineH;

//    m_legend->SetY2(y);
//    m_legend->Draw();
//    m_legend = 0; // Deleted together with TPad.

}

void
FWConvTrackHitsDetailView::makeLegend( void )
{
   m_legend = new TLegend( 0.01, 0.01, 0.99, 0.99, 0, "NDC" );
   m_legend->SetFillColor(kWhite);
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
FWConvTrackHitsDetailView::addTrackerHits3D( std::vector<TVector3> &points, class TEveElementList *tList, Color_t color, int size ) 
{
   // !AT this is  detail view specific, should move to track hits
   // detail view

   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetMarkerSize(size);
   pointSet->SetMarkerStyle(4);
   pointSet->SetPickable(kTRUE);
   pointSet->SetTitle("Pixel Hits");
   pointSet->SetMarkerColor(color);
		
   for( std::vector<TVector3>::const_iterator it = points.begin(), itEnd = points.end(); it != itEnd; ++it) {
      pointSet->SetNextPoint(it->x(), it->y(), it->z());
   }
   tList->AddElement(pointSet);
}

void
FWConvTrackHitsDetailView::addHits( const reco::Track& track,
				const FWEventItem* iItem,
				TEveElement* trkList,
				bool addNearbyHits )
{
   std::vector<TVector3> pixelPoints;
   fireworks::pushPixelHits( pixelPoints, *iItem, track );
   TEveElementList* pixels = new TEveElementList( "Pixels" );
   trkList->AddElement( pixels );
   if( addNearbyHits )
   {
      // get the extra hits
      std::vector<TVector3> pixelExtraPoints;
      fireworks::pushNearbyPixelHits( pixelExtraPoints, *iItem, track );
      // draw first the others
      addTrackerHits3D( pixelExtraPoints, pixels, kRed, 1 );
      // then the good ones, so they're on top
      addTrackerHits3D( pixelPoints, pixels, kGreen, 1 );
   }
   else
   {
      // just add those points with the default color
      addTrackerHits3D( pixelPoints, pixels, iItem->defaultDisplayProperties().color(), 1 );
   }

   // strips
   TEveElementList* strips = new TEveElementList( "Strips" );
   trkList->AddElement( strips );
   fireworks::addSiStripClusters( iItem, track, strips, addNearbyHits, false );
}

//______________________________________________________________________________

void
FWConvTrackHitsDetailView::addModules( const reco::Track& track,
				   const FWEventItem* iItem,
				   TEveElement* trkList,
				   bool addLostHits )
{
   std::set<unsigned int> ids;
   for( trackingRecHit_iterator recIt = track.recHitsBegin(), recItEnd = track.recHitsEnd();
	recIt != recItEnd; ++recIt )
   {
      DetId detid = (*recIt)->geographicalId();
      if( !addLostHits && !(*recIt)->isValid()) continue;
      if( detid.rawId() != 0 )
      {
	 TString name("");
	 switch( detid.det())
	 {
	 case DetId::Tracker:
	    switch( detid.subdetId())
	    {
	    case SiStripDetId::TIB:
	       name = "TIB ";
	       break;
	    case SiStripDetId::TOB:
	       name = "TOB ";
	       break;
	    case SiStripDetId::TID:
	       name = "TID ";
	       break;
	    case SiStripDetId::TEC:
	       name = "TEC ";
	       break;
	    case PixelSubdetector::PixelBarrel:
	       name = "Pixel Barrel ";
	       break;
	    case PixelSubdetector::PixelEndcap:
	       name = "Pixel Endcap ";
	    default:
	       break;
	    }
	    break;
	    
	 case DetId::Muon:
	    switch( detid.subdetId())
	    {
	    case MuonSubdetId::DT:
	       name = "DT";
	       detid = DetId( DTChamberId( detid )); // get rid of layer bits
	       break;
	    case MuonSubdetId::CSC:
	       name = "CSC";
	       break;
	    case MuonSubdetId::RPC:
	       name = "RPC";
	       break;
	    default:
	       break;
	    }
	    break;
	 default:
	    break;
	 }
	 if( ! ids.insert( detid.rawId()).second ) continue;
	 if( iItem->getGeom())
	 {
	    TEveGeoShape* shape = iItem->getGeom()->getEveShape( detid );
	    if( 0 != shape )
	    {
	       shape->SetMainTransparency( 65 );
	       shape->SetPickable( kTRUE );
	       switch(( *recIt )->type())
	       {
	       case TrackingRecHit::valid:
		  shape->SetMainColor( iItem->defaultDisplayProperties().color());
		  break;
	       case TrackingRecHit::missing:
		  name += "LOST ";
		  shape->SetMainColor( kRed );
		  break;
	       case TrackingRecHit::inactive:
		  name += "INACTIVE ";
		  shape->SetMainColor( 28 );
		  break;
	       case TrackingRecHit::bad:
		  name += "BAD ";
		  shape->SetMainColor( 218 );
		  break;
	       }
	       shape->SetTitle( name + ULong_t( detid.rawId()));
	       trkList->AddElement( shape );
	    }
	    else
	    {
	       fwLog( fwlog::kInfo ) <<  "Failed to get shape extract for a tracking rec hit: "
				     << "\n" << fireworks::info( detid ) << std::endl;
	    }
	 }
      }
   }
}

void
FWConvTrackHitsDetailView::rnrLabels()
{
   m_moduleLabels->SetRnrChildren(!m_moduleLabels->GetRnrChildren());
   gEve->Redraw3D();
}

REGISTER_FWDETAILVIEW(FWConvTrackHitsDetailView, Conversion);
