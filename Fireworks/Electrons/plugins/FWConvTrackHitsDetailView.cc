
#define protected public
#include "TGLViewer.h" // access to over-all bounding box
#include "TEveCalo.h" // workaround for TEveCalo3D bounding box
#undef protected
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
#include "TCanvas.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TG3DLine.h"
#include "TEveCaloData.h"

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
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Core/interface/FWBeamSpot.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"

#include "Fireworks/Electrons/plugins/FWConvTrackHitsDetailView.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

namespace {
void (FWConvTrackHitsDetailView::*foo)();
}

FWConvTrackHitsDetailView::FWConvTrackHitsDetailView ():
  m_modules(0),
  m_moduleLabels(0),
  m_hits(0),
  m_legend(0),
  m_orthographic(false)
{
}

FWConvTrackHitsDetailView::~FWConvTrackHitsDetailView ()
{
}

namespace {
void setCameraInit(TGLViewer* v, TGLViewer::ECameraType type, const TEveVectorD b1, TEveVectorD b3, TEveVector center )
{ 
   TGLCamera& cam = v->RefCamera(type); 
   TGLMatrix& trans = cam.RefCamBase();

   trans.Set(trans.GetTranslation(), b3.Arr(), b1.Arr());
   cam.Setup(v->fOverallBoundingBox, kTRUE);
  
   cam.SetExternalCenter(true);
   cam.SetCenterVec(center.fX, center.fY, center.fZ);
}
   
   /*
    // alternative to setCameraInit
void setCamera(TGLViewer* v, TGLViewer::ECameraType type, const TEveVectorD b1, TEveVectorD b2,  TEveVectorD b3, TEveVector center )
{
   // b1 = fwd, b2 = lft, b3 = up

   TGLCamera& cam = v->RefCamera(type); 
   TGLMatrix& trans = cam.RefCamBase();
   
   trans.SetBaseVec(1, b1.Arr());
   trans.SetBaseVec(2, b2.Arr());
   trans.SetBaseVec(3, b3.Arr());

   cam.SetExternalCenter(true);
   cam.SetCenterVec(center.fX, center.fY, center.fZ);
}
    */
}
void
FWConvTrackHitsDetailView::build (const FWModelId &id, const reco::Conversion* conv)
{      
   if (conv->nTracks()<2) return;
   const reco::Track* track0 = conv->tracks().at(0).get();
   const reco::Track* track1 = conv->tracks().at(1).get();
   
   m_guiFrame->AddFrame(new TGLabel(m_guiFrame, "Camera Views:"), new TGLayoutHints(kLHintsLeft, 2, 2, 2, 2)); 

   {
      TGHorizontalFrame* f = new TGHorizontalFrame(m_guiFrame);
      m_guiFrame->AddFrame(f,  new TGLayoutHints(kLHintsExpandX, 2, 0, 0, 0));
      
      {
         CSGAction* action = new CSGAction(this, "Top");
        TGTextButton* b = new TGTextButton(f,  action->getName().c_str());
         f->AddFrame(b, new TGLayoutHints( kLHintsExpandX));
         TQObject::Connect(b, "Clicked()", "CSGAction", action, "activate()");
         b->SetToolTipText("plane normal: track0 x track1");
         action->activated.connect(sigc::mem_fun(this, &FWConvTrackHitsDetailView::camera1Callback));
      }

      {
         CSGAction* action = new CSGAction(this, "Side");
         TGTextButton* b = new TGTextButton(f,  action->getName().c_str());
         f->AddFrame(b, new TGLayoutHints( kLHintsExpandX));
         TQObject::Connect(b, "Clicked()", "CSGAction", action, "activate()");
         b->SetToolTipText("left fir: track1");
         action->activated.connect(sigc::mem_fun(this, &FWConvTrackHitsDetailView::camera2Callback));
      } 
      {
         CSGAction* action = new CSGAction(this, "Front");
         TGTextButton* b = new TGTextButton(f,  action->getName().c_str());
         f->AddFrame(b, new TGLayoutHints( kLHintsExpandX));
         TQObject::Connect(b, "Clicked()", "CSGAction", action, "activate()");
         action->activated.connect(sigc::mem_fun(this, &FWConvTrackHitsDetailView::camera3Callback));
         b->SetToolTipText("plane normal: track1 ");

      }
   }
   
   {
      m_camTypeAction = new CSGAction(this, " Set Ortographic Projection ");
      m_camTypeAction->createTextButton(m_guiFrame, new TGLayoutHints( kLHintsExpandX, 2, 0, 1, 4));
      m_camTypeAction->activated.connect(sigc::mem_fun(this, &FWConvTrackHitsDetailView::switchProjection));
   }
   
   {
      CSGAction* action = new CSGAction(this, "Draw Module");
      TGCheckButton* b = new TGCheckButton(m_guiFrame, action->getName().c_str() );
      b->SetState(kButtonDown, false);
      m_guiFrame->AddFrame(b, new TGLayoutHints( kLHintsNormal, 2, 3, 1, 4));
      TQObject::Connect(b, "Clicked()", "CSGAction", action, "activate()");
      action->activated.connect(sigc::mem_fun(this, &FWConvTrackHitsDetailView::rnrModules));
   }
   {
      CSGAction* action = new CSGAction(this, "Draw Hits");
      TGCheckButton* b = new TGCheckButton(m_guiFrame, action->getName().c_str() );
      b->SetState(kButtonDown, false);
      m_guiFrame->AddFrame(b, new TGLayoutHints( kLHintsNormal, 2, 3, 1, 4));
      TQObject::Connect(b, "Clicked()", "CSGAction", action, "activate()");
      action->activated.connect(sigc::mem_fun(this, &FWConvTrackHitsDetailView::rnrHits));
   }
   
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
     // gs->SetMainTransparency(75);
     // gs->SetPickable(kFALSE);

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
   
   // first-vertex style
   prop->SetRnrFV(kTRUE);
   prop->RefFVAtt().SetMarkerColor(id.item()->defaultDisplayProperties().color());
   prop->RefFVAtt().SetMarkerSize(0.8);

   // path-mark style
   prop->RefPMAtt().SetMarkerSize(0.5);
   prop->RefPMAtt().SetMarkerColor(id.item()->defaultDisplayProperties().color());

   TEveTrack* trk1 = fireworks::prepareTrack( *track1, prop );
   trk1->MakeTrack();
   trk1->SetLineWidth(2);
   trk1->SetTitle( "Track 1 and its ref states" );
   trk1->SetMainColor(id.item()->defaultDisplayProperties().color());
   trk1->SetLineStyle(7);
   m_eveScene->AddElement(trk1);

   TEveTrack* trk0 = fireworks::prepareTrack( *track0, prop );
   trk0->MakeTrack();
   trk0->SetLineWidth(2);
   trk0->SetTitle( "Track 0 and its ref states" );
   trk0->SetMainColor(id.item()->defaultDisplayProperties().color());
   m_eveScene->AddElement(trk0);
   
   // create TEveCalo3D object, fix bounding box
   {
      float phi = conv->pairMomentum().phi();
      float eta = conv->pairMomentum().eta();
      
      FWECALDetailViewBuilder caloBld( id.item()->getEvent(), id.item()->getGeom(), eta, phi, 30);
      TEveCaloData* data = caloBld.buildCaloData(false);
       // AMT!!! this is mempry leak, check why it needs to be added
      TEveCalo3D* calo3d = new TEveCalo3D(data);
      gEve->AddElement(data);
      calo3d->SetBarrelRadius(129.00);
      calo3d->SetEndCapPos(268.36);
      
      float theta = TEveCaloData::EtaToTheta(eta);
      float eps = data->GetMaxVal(true) * calo3d->GetValToHeight();
      if (TMath::Abs(eta) < calo3d->GetTransitionEta())
      {
        // printf("barrel\n");
         float x =   calo3d->GetBarrelRadius() * TMath::Cos(phi);
         float y =   calo3d->GetBarrelRadius() * TMath::Sin(phi);
         float z =   calo3d->GetBarrelRadius() / TMath::Tan(theta); 
         
         calo3d->BBoxZero(eps, x, y, z);
      }
      else
      {
       //  printf("endcap\n");
         float z  = TMath::Sign(calo3d->GetEndCapPos(), eta);
         float r =   z*TMath::Tan(theta);
         calo3d->BBoxZero(eps, r* TMath::Cos(phi), r*TMath::Sin(phi), z);
      }
      m_eveScene->AddElement(calo3d);
   }
   
   //  base vectors
   TEveVectorD fwd = trk1->GetMomentum().Cross(trk0->GetMomentum());
   fwd.Normalize();
   TEveVectorD lft =trk1->GetMomentum();
   lft.Normalize();
   TEveVectorD up = lft.Cross(fwd);

   
   TEveVectorD c  = ( trk1->GetVertex() + trk0->GetVertex()) *0.5;
   if (1)
   { 
      setCameraInit(viewerGL(),TGLViewer::kCameraPerspXOZ, fwd, up, c); //default
      setCameraInit(viewerGL(),TGLViewer::kCameraPerspYOZ, up,  lft,c);
      setCameraInit(viewerGL(),TGLViewer::kCameraPerspXOY, lft, fwd,c);

      setCameraInit(viewerGL(),TGLViewer::kCameraOrthoXOY, fwd,  up,c);
      setCameraInit(viewerGL(),TGLViewer::kCameraOrthoXOZ, up, lft,c);
      setCameraInit(viewerGL(),TGLViewer::kCameraOrthoZOY, lft, fwd,c); 
   }
   {
      Float_t sfac = 100;
      fwd *= sfac;
      lft *= sfac;
      up  *= sfac;
      int transp = 90;
      {
         TEveStraightLineSet* bls = new TEveStraightLineSet("base1");
         bls->AddLine(c, fwd + c);
         bls->SetMainColor(kBlue);
         bls->SetMainTransparency(transp);
         bls->SetPickable(false);
         m_eveScene->AddElement(bls);
      }

      {
         TEveStraightLineSet* bls = new TEveStraightLineSet("base2");
         bls->AddLine(c, lft + c);
         bls->SetMainColor(kBlue);
         bls->SetMainTransparency(transp);
         bls->SetPickable(false);
         m_eveScene->AddElement(bls);
      }

      {
         TEveStraightLineSet* bls = new TEveStraightLineSet("base3");
         bls->AddLine(c, up + c);
         bls->SetMainColor(kBlue);
         bls->SetMainTransparency(transp);
         bls->SetPickable(false);
         m_eveScene->AddElement(bls);
      }
   }
   {
      TEveStraightLineSet* bls = new TEveStraightLineSet("Photon", "Photon");
      FWBeamSpot* bs =  context().getBeamSpot();
      bls->AddLine(c.fX, c.fY, c.fZ, bs->x0(), bs->y0(), bs->z0());
      bls->SetMainColor(id.item()->defaultDisplayProperties().color());
      bls->SetLineStyle(3);
      m_eveScene->AddElement(bls);
   }
   
   viewerGL()->SetStyle(TGLRnrCtx::kOutline);
   viewerGL()->ResetCamerasAfterNextUpdate();
   viewerGL()->UpdateScene(kFALSE);
   gEve->Redraw3D();
   
   setTextInfo(id, conv);
   foo = &FWConvTrackHitsDetailView::camera1Callback; 
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
   const reco::HitPattern &hp0 = track0->hitPattern();
   int nvalid_tk0 = 0, ninvalid_tk0 = 0, npix_tk0 = 0, nstrip_tk0 = 0;
   for(int i_tk0 = 0; i_tk0 < hp0.numberOfHits(reco::HitPattern::TRACK_HITS); i_tk0++) {
       uint32_t hit = hp0.getHitPattern(reco::HitPattern::TRACK_HITS, i_tk0);
       if(reco::HitPattern::validHitFilter(hit)) {
           nvalid_tk0++;
           if (reco::HitPattern::pixelHitFilter(hit)) npix_tk0++;
           else if (reco::HitPattern::stripHitFilter(hit)) nstrip_tk0++;
       } else ninvalid_tk0++;
   }
   latex->DrawLatex( x, y,  Form( "valid hits: %i (pix. %i, str. %i)", nvalid_tk0, npix_tk0, nstrip_tk0) );
   y -= lineH;
   latex->DrawLatex( x, y,  Form( "invalid: %i", ninvalid_tk0) );
   y -= lineH;

   int npix_mhi_tk0 = 0, nstrip_mhi_tk0 = 0;
   for(int i_mhi_tk0 = 0; i_mhi_tk0 < hp0.numberOfHits(reco::HitPattern::MISSING_INNER_HITS); i_mhi_tk0++) {
       uint32_t hit = hp0.getHitPattern(reco::HitPattern::MISSING_INNER_HITS, i_mhi_tk0);
       if (reco::HitPattern::pixelHitFilter(hit)) npix_mhi_tk0++;
       else if (reco::HitPattern::stripHitFilter(hit)) nstrip_mhi_tk0++;
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

   const reco::HitPattern &hp1 = track1->hitPattern();
   int nvalid_tk1 = 0, ninvalid_tk1 = 0, npix_tk1 = 0, nstrip_tk1 = 0;
   for(int i_tk1 = 0; i_tk1 < hp1.numberOfHits(reco::HitPattern::TRACK_HITS); i_tk1++) {
       uint32_t hit = hp1.getHitPattern(reco::HitPattern::TRACK_HITS, i_tk1);
       if(reco::HitPattern::validHitFilter(hit)) {
           nvalid_tk1++;
           if (reco::HitPattern::pixelHitFilter(hit)) npix_tk1++;
           else if (reco::HitPattern::stripHitFilter(hit)) nstrip_tk1++;
       } else ninvalid_tk1++;
   }
   latex->DrawLatex( x, y,  Form( "valid hits: %i (pix. %i, str. %i)", nvalid_tk1, npix_tk1, nstrip_tk1) );
   y -= lineH;
   latex->DrawLatex( x, y,  Form( "invalid: %i", ninvalid_tk1) );
   y -= lineH;

   int npix_mhi_tk1 = 0, nstrip_mhi_tk1 = 0;
   for(int i_mhi_tk1 = 0; i_mhi_tk1 < hp1.numberOfHits(reco::HitPattern::MISSING_INNER_HITS); i_mhi_tk1++) {
       uint32_t hit = hp1.getHitPattern(reco::HitPattern::MISSING_INNER_HITS, i_mhi_tk1);
       if (reco::HitPattern::pixelHitFilter(hit)) npix_mhi_tk1++;
       else if (reco::HitPattern::stripHitFilter(hit)) nstrip_mhi_tk1++;
   }
   latex->DrawLatex( x, y,  Form("miss. inner hits: pix. %i, str. %i", npix_mhi_tk1, nstrip_mhi_tk1) );
   y -= lineH;
   y -= lineH;
}


void
FWConvTrackHitsDetailView::addTrackerHits3D( std::vector<TVector3> &points, class TEveElementList *tList, Color_t color, int size ) 
{
   // !AMT this is  detail view specific, should move to track hits
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
	       case TrackingRecHit::missing_inner:
	       case TrackingRecHit::missing_outer:
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

//______________________________________________________________________________
void
FWConvTrackHitsDetailView::pickCameraCenter()
{
   viewerGL()->PickCameraCenter();
   viewerGL()->SetDrawCameraCenter(kTRUE);
}

void
FWConvTrackHitsDetailView::switchProjection()
{   
   m_orthographic = !m_orthographic;
   m_camTypeAction->setName(m_orthographic ? "Set Perspective Projection" : "Set Orthographic Projection");
   
   (this->*foo) ();
   //printf("current isOrthographic : %d \n",  viewerGL()->CurrentCamera().IsOrthographic());
}


void
FWConvTrackHitsDetailView::rnrLabels()
{
   m_moduleLabels->SetRnrChildren(!m_moduleLabels->GetRnrChildren());
   gEve->Redraw3D();
}

void
FWConvTrackHitsDetailView::rnrModules()
{
   m_modules->SetRnrChildren(!m_modules->GetRnrChildren());
   gEve->Redraw3D();
}

void
FWConvTrackHitsDetailView::rnrHits()
{
   m_hits->SetRnrChildren(!m_hits->GetRnrChildren());
   gEve->Redraw3D();
}


void FWConvTrackHitsDetailView::camera1Callback()
{
   foo = &FWConvTrackHitsDetailView::camera1Callback; 
   
   viewerGL()->SetCurrentCamera(m_orthographic ? TGLViewer::kCameraOrthoXOY : TGLViewer::kCameraPerspXOZ);
   viewerGL()->ResetCurrentCamera();
   viewerGL()->RequestDraw();
}

void FWConvTrackHitsDetailView::camera2Callback()
{
   foo = &FWConvTrackHitsDetailView::camera2Callback; 
   
   viewerGL()->SetCurrentCamera( m_orthographic ? TGLViewer::kCameraOrthoXOZ : TGLViewer::kCameraPerspYOZ);
   viewerGL()->ResetCurrentCamera();
   viewerGL()->RequestDraw();

}

void FWConvTrackHitsDetailView::camera3Callback()
{
   foo = &FWConvTrackHitsDetailView::camera3Callback; 

   viewerGL()->SetCurrentCamera(m_orthographic ? TGLViewer::kCameraOrthoZOY : TGLViewer::kCameraPerspXOY);
   viewerGL()->ResetCurrentCamera();
   viewerGL()->RequestDraw();

}


REGISTER_FWDETAILVIEW(FWConvTrackHitsDetailView, Conversion, ecalRecHit);
REGISTER_FWDETAILVIEW(FWConvTrackHitsDetailView, Conversion, reducedEcalRecHitsEB);
