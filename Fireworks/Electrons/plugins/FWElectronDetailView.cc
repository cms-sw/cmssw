// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWElectronDetailView
// $Id: FWElectronDetailView.cc,v 1.40 2009/10/08 18:23:14 amraktad Exp $
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
#include "Fireworks/Electrons/plugins/FWElectronDetailView.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"

//
// constructors and destructor
//
FWElectronDetailView::FWElectronDetailView():
m_viewer(0)
{
}

FWElectronDetailView::~FWElectronDetailView()
{ 
   getEveWindow()->DestroyWindow();
}

//
// member functions
//
void FWElectronDetailView::build(const FWModelId &id, const reco::GsfElectron* iElectron, TEveWindowSlot* base)
{
   if(0==iElectron) return;
   TEveWindowPack* eveWindow = base->MakePack();
   eveWindow->SetShowTitleBar(kFALSE);

   TEveWindow* ew(0);
   TEveWindowSlot* slot(0);
   TEveScene*      scene(0);
   TEveViewer*     viewer(0);
   TGVerticalFrame* ediFrame(0);

   ////////////////////////////////////////////////////////////////////////
   //                              Sub-view 1
   ///////////////////////////////////////////////////////////////////////

   // prepare window
   slot = eveWindow->NewSlot();
   ew = FWDetailViewBase::makePackViewer(slot, ediFrame, viewer, scene);
   ew->SetElementName("ECAL based view");
   FWDetailViewBase::setEveWindow(ew);
   
   // build ECAL objects
   FWECALDetailViewBuilder builder(id.item()->getEvent(), id.item()->getGeom(),
				   iElectron->caloPosition().eta(), iElectron->caloPosition().phi(), 25);
   builder.showSuperClusters(kGreen+2, kGreen+4);
   if ( iElectron->superCluster().isAvailable() )
     builder.showSuperCluster(*(iElectron->superCluster()), kYellow);
   TEveCaloLego* lego = builder.build();
   scene->AddElement(lego);
   
   // add Electron specific details
   addTrackPointsInCaloData( iElectron, lego);
   drawCrossHair(iElectron, lego, scene);
   addInfo(iElectron, scene);

   // draw legend in latex
   TRootEmbeddedCanvas* ec = new TRootEmbeddedCanvas("Embeddedcanvas", ediFrame, 100, 100, 0);
   ediFrame->AddFrame(ec, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   ediFrame->MapSubwindows();
   ediFrame->Layout();
   ec->GetCanvas()->SetBorderMode(0);
   makeLegend(iElectron, id, ec->GetCanvas());
   
   // draw axis at the window corners
   // std::cout << "TEveViewer: " << viewer << std::endl;
   m_viewer =  viewer->GetGLViewer();
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   overlay->SetShowPlane(kFALSE);
   overlay->SetShowPerspective(kFALSE);
   overlay->SetCaloLego(lego);
   overlay->SetShowScales(1); // temporary
   m_viewer->AddOverlayElement(overlay);

   // set event handler and flip camera to top view at beginning
   m_viewer->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TEveLegoEventHandler* eh = 
      new TEveLegoEventHandler((TGWindow*)m_viewer->GetGLWidget(), (TObject*)m_viewer, lego);
   m_viewer->SetEventHandler(eh);
   m_viewer->UpdateScene();
   m_viewer->CurrentCamera().Reset();

   scene->Repaint(true);

   m_viewer->RequestDraw(TGLRnrCtx::kLODHigh);
   gEve->Redraw3D();
   
   ////////////////////////////////////////////////////////////////////////
   //                              Sub-view 2
   ///////////////////////////////////////////////////////////////////////
   // slot = eveWindow->NewSlot();
   // ew = FWDetailViewBase::makePackViewer(slot, ediFrame, viewer, scene);
   // ew->SetElementName("View C");

   // eveWindow->GetTab()->SetTab(1);
}


math::XYZPoint FWElectronDetailView::trackPositionAtCalo (const reco::GsfElectron &t)
{
   return t.TrackPositionAtCalo();
}

double FWElectronDetailView::deltaEtaSuperClusterTrackAtVtx (const reco::GsfElectron &t)
{
   return t.deltaEtaSuperClusterTrackAtVtx();
}

double FWElectronDetailView::deltaPhiSuperClusterTrackAtVtx (const reco::GsfElectron &t)
{
   return t.deltaPhiSuperClusterTrackAtVtx();
}

void
FWElectronDetailView::makeLegend(const reco::GsfElectron *electron, 
				 const FWModelId& id,
				 TCanvas* textCanvas)
{
   textCanvas->cd();

   TLatex* latex = new TLatex(0.02, 0.970, "");
   const double textsize(0.05);
   latex->SetTextSize(2*textsize);

   float_t x = 0.02;
   float_t x2 = 0.52;
   float   y = 0.95;
   float fontsize = latex->GetTextSize()*0.6;
   
   latex->DrawLatex(x, y, id.item()->modelName(id.index()).c_str() );
   y -= fontsize;
   latex->SetTextSize(textsize);
   fontsize = latex->GetTextSize()*0.6;

   latex->DrawLatex(x, y, Form(" E_{T} = %.1f GeV, #eta = %0.2f, #varphi = %0.2f", 
			       electron->et(), electron->eta(), electron->phi()) );
   y -= fontsize;
   // summary
   if (electron->charge() > 0)
     latex->DrawLatex(x, y, " charge = +1");
   else 
     latex->DrawLatex(x, y, " charge = -1");
   y -= fontsize;

   // delta phi/eta in
   latex->DrawLatex(x, y, "SuperCluster vs inner state extrapolation");
   y -= fontsize;
   latex->DrawLatex(x, y,  Form(" #Delta#eta_{in} = %.3f",
				electron->deltaEtaSuperClusterTrackAtVtx()) );
   latex->DrawLatex(x2, y, Form("#Delta#varphi_{in} = %.3f",
				electron->deltaPhiSuperClusterTrackAtVtx()) );
   y -= fontsize;
   
   // delta phi/eta out
   latex->DrawLatex(x, y, "SeedCluster vs outer state extrapolation");
   y -= fontsize;
   char dout[128];
   sprintf(dout, " #Delta#eta_{out} = %.3f",
	   electron->deltaEtaSeedClusterTrackAtCalo());
   latex->DrawLatex(x, y, dout);
   sprintf(dout, " #Delta#varphi_{out} = %.3f",
	   electron->deltaPhiSeedClusterTrackAtCalo());
   latex->DrawLatex(x2, y, dout);
   y -= 2*fontsize;
   
   // legend
   /* it should be clear from the tool tip pop-up what a user is looking at
   latex->SetTextSize(1.5*textsize);
   latex->DrawLatex(x, y, "Legend:"); y -= fontsize*1.5;
   latex->SetTextSize(textsize);
   latex->DrawLatex(x, y, " #color[2]{#bullet} seed cluster centroid");
   y -= fontsize;
   latex->DrawLatex(x, y, " #color[4]{#bullet} supercluster centroid");
   y -= fontsize;
   // latex->DrawLatex(x, y, "#color[618]{#Box} seed cluster");
   // y -= fontsize;
   // latex->DrawLatex(x, y, "#color[608]{#Box} other clusters"); // kCyan+1
   // eta, phi axis or x, y axis?
   // y -= fontsize;

   latex->DrawLatex(x, y, "track position at calo:");
   y -= fontsize;
   latex->DrawLatex(x, y, " #color[2]{+} extrapolating from innermost state");
   y -= fontsize;
   latex->DrawLatex(x, y, " #color[4]{+} extrapolating from outermost state");
   y -= fontsize;
    */
}

void
FWElectronDetailView::drawCrossHair (const reco::GsfElectron* i, TEveCaloLego *lego, TEveElementList* tList)
{
    unsigned int subdetId(0);
    if ( ! i->superCluster()->seed()->hitsAndFractions().empty() )
      subdetId = i->superCluster()->seed()->hitsAndFractions().front().first.subdetId();

   double ymax = lego->GetPhiMax();
   double ymin = lego->GetPhiMin();
   double xmax = lego->GetEtaMax();
   double xmin = lego->GetEtaMin();

   // draw crosshairs for track intersections

   {
      const double eta = i->superCluster()->seed()->position().eta() -
         i->deltaEtaSeedClusterTrackAtCalo();
      const double phi = i->superCluster()->seed()->position().phi() -
         i->deltaPhiSeedClusterTrackAtCalo();

      TEveStraightLineSet *trackpositionAtCalo = new TEveStraightLineSet("sc trackpositionAtCalo");
      trackpositionAtCalo->SetPickable(kTRUE);
      trackpositionAtCalo->SetTitle("Track position at Calo propagating from the outermost state");
      if (subdetId == EcalBarrel)
      {
         trackpositionAtCalo->AddLine(eta, ymin, 0, eta, ymax, 0);
         trackpositionAtCalo->AddLine(xmin, phi, 0, xmax, phi, 0);
      }
      else if (subdetId == EcalEndcap)
      {
         TVector3 pos;
         pos.SetPtEtaPhi(i->superCluster()->seed()->position().rho(), eta, phi);
         trackpositionAtCalo->AddLine(pos.X(), ymin, 0, pos.X(), ymax, 0);
         trackpositionAtCalo->AddLine(xmin, pos.Y(), 0, xmax,pos.Y(),0);
      }
      trackpositionAtCalo->SetDepthTest(kFALSE);
      trackpositionAtCalo->SetLineColor(kBlue);
      tList->AddElement(trackpositionAtCalo);
   }
   //
   // pin position
   //
   {
      TEveStraightLineSet *pinposition = new TEveStraightLineSet("pin position");
      pinposition->SetPickable(kTRUE);
      pinposition->SetTitle("Track position at Calo propagating from the innermost state");
      Double_t eta = i->caloPosition().eta() - deltaEtaSuperClusterTrackAtVtx(*i);
      Double_t phi = i->caloPosition().phi() - deltaPhiSuperClusterTrackAtVtx(*i);

      if (subdetId == EcalBarrel)
      {
         pinposition->AddLine(eta, ymax, 0, eta, ymin, 0);
         pinposition->AddLine(xmin, phi, 0, xmax, phi, 0);
      }
      else if (subdetId == EcalEndcap)
      {
         TVector3 pos;
         pos.SetPtEtaPhi(i->caloPosition().rho(), eta, phi);
         pinposition->AddLine(pos.X(),ymin, 0, pos.X(), ymax, 0);
         pinposition->AddLine(xmin, pos.Y(), 0, xmax, pos.Y(), 0);
      }
      pinposition->SetDepthTest(kFALSE);
      pinposition->SetLineColor(kRed);
      tList->AddElement(pinposition);
   }
/*
   printf("TrackPositionAtCalo: %f %f\n",
          trackPositionAtCalo(*i).eta(), trackPositionAtCalo(*i).phi());
   printf("TrackPositionAtCalo: %f %f\n",
          i->superCluster()->seed()->position().eta() -
          i->deltaEtaSeedClusterTrackAtCalo(),
          i->superCluster()->seed()->position().phi() -
          i->deltaPhiSeedClusterTrackAtCalo());
   printf("TrackPositionInner: %f %f\n",
          i->caloPosition().eta() - deltaEtaSuperClusterTrackAtVtx(*i),
          i->caloPosition().phi() - deltaPhiSuperClusterTrackAtVtx(*i));
   printf("calo position %f, deltaEta %f, track position %f\n",
          i->caloPosition().eta(),
          deltaEtaSuperClusterTrackAtVtx(*i),
          trackPositionAtCalo(*i).eta());
 */
}

Bool_t FWElectronDetailView::checkRange(Double_t &em, Double_t& eM, Double_t &pm, Double_t& pM,
                                       Double_t eta, Double_t phi)
{
   Bool_t changed = kFALSE;

   //check eta
   if (eta < em)
   {
      em = eta;
      changed = kTRUE;
   }
   else if (eta > eM)
   {
      eM = eta;
      changed = kTRUE;
   }

   // check phi
   if (phi < pm)
   {
      pm = phi;
      changed = kTRUE;
   }
   else if (phi > pM)
   {
      pM = phi;
      changed = kTRUE;
   }
   return changed;
}

void
FWElectronDetailView::addTrackPointsInCaloData (const reco::GsfElectron *i, TEveCaloLego* lego)
{
  unsigned int subdetId(0);
  if ( ! i->superCluster()->seed()->hitsAndFractions().empty() )
    subdetId = i->superCluster()->seed()->hitsAndFractions().front().first.subdetId();

   TEveCaloDataVec* data = (TEveCaloDataVec*)lego->GetData();
   Double_t em, eM, pm, pM;
   data->GetEtaLimits(em, eM);
   data->GetPhiLimits(pm, pM);

   Bool_t changed = kFALSE;
   // add cells in third layer if necessary

   //   trackpositionAtCalo
   {
      double eta = i->superCluster()->seed()->position().eta() -
         i->deltaEtaSeedClusterTrackAtCalo();
      double phi = i->superCluster()->seed()->position().phi() -
         i->deltaPhiSeedClusterTrackAtCalo();

      if (subdetId == EcalBarrel)
      {
         if (checkRange(em, eM, pm, pM, eta, phi))
            changed = kTRUE;
      }
      else if (subdetId == EcalEndcap) {
         TVector3 pos;
         pos.SetPtEtaPhi(i->superCluster()->seed()->position().rho(),eta, phi);
         if (checkRange(em, eM, pm, pM, pos.X(), pos.Y()))
            changed = kTRUE;

      }
   }
   // pinposition
   {
      double eta = i->caloPosition().eta() - deltaEtaSuperClusterTrackAtVtx(*i);
      double phi = i->caloPosition().phi() - deltaPhiSuperClusterTrackAtVtx(*i);
      if (subdetId == EcalBarrel)
      {
         if (checkRange(em, eM, pm, pM, eta, phi))
            changed = kTRUE;
      }
      else if (subdetId == EcalEndcap) {
         TVector3 pos;
         pos.SetPtEtaPhi(i->caloPosition().rho(), eta, phi);
         if (checkRange(em, eM, pm, pM, pos.X(), pos.Y()))
            changed = kTRUE;
      }
   }
   if (changed)
   {
      data->AddTower(em, eM, pm, pM);
      data->FillSlice(2, 0);   data->DataChanged();

      lego->ComputeBBox();
      Double_t legoScale = ((eM - em) < (pM - pm)) ? (eM - em) : (pM - pm);
      lego->InitMainTrans();
      lego->RefMainTrans().SetScale(legoScale, legoScale, legoScale*0.5);
      lego->RefMainTrans().SetPos((eM+em)*0.5, (pM+pm)*0.5, 0);
      lego->ElementChanged(true);
   }
}

void
FWElectronDetailView::addInfo(const reco::GsfElectron *i, TEveElementList* tList)
{
   unsigned int subdetId(0);
   if ( ! i->superCluster()->seed()->hitsAndFractions().empty() )
     subdetId = i->superCluster()->seed()->hitsAndFractions().front().first.subdetId();
  
   // points for centroids
   Double_t x(0), y(0), z(0);
   TEvePointSet *scposition = new TEvePointSet("sc position");
   scposition->SetPickable(kTRUE);
   scposition->SetTitle("Super cluster centroid");
   if (subdetId == EcalBarrel) {
      x = i->caloPosition().eta();
      y = i->caloPosition().phi();
   } else if (subdetId == EcalEndcap) {
      x = i->caloPosition().x();
      y = i->caloPosition().y();
   }
   scposition->SetNextPoint(x,y,z);
   scposition->SetMarkerSize(1);
   scposition->SetMarkerStyle(4);
   scposition->SetMarkerColor(kBlue);
   tList->AddElement(scposition);

   // points for seed position
   TEvePointSet *seedposition = new TEvePointSet("seed position");
   seedposition->SetTitle("Seed cluster centroid");
   seedposition->SetPickable(kTRUE);
   if (subdetId == EcalBarrel) {
      x  = i->superCluster()->seed()->position().eta();
      y  = i->superCluster()->seed()->position().phi();
      seedposition->SetMarkerSize(0.01);
   } else if (subdetId == EcalEndcap) {
      x  = i->superCluster()->seed()->position().x();
      y  = i->superCluster()->seed()->position().y();
      seedposition->SetMarkerSize(1);
   }
   seedposition->SetNextPoint(x, y, z);
   seedposition->SetMarkerStyle(2);
   seedposition->SetMarkerColor(kRed);
   tList->AddElement(seedposition);
   
   // electron direction (show it if it's within
   // the area of interest)
   if ( fabs(i->phi()-i->caloPosition().phi())< 25*0.0172 &&
	fabs(i->eta()-i->caloPosition().eta())< 25*0.0172 )
     {
	TEvePointSet *eldirection = new TEvePointSet("seed position");
	eldirection->SetTitle("Electron direction at vertex");
	eldirection->SetPickable(kTRUE);
	eldirection->SetMarkerStyle(2);
	if (subdetId == EcalBarrel) {
	   eldirection->SetNextPoint(i->eta(), i->phi(), z);
	   eldirection->SetMarkerSize(0.01);
	}else{
	   eldirection->SetNextPoint(310*fabs(tan(i->theta()))*cos(i->phi()), 
				     310*fabs(tan(i->theta()))*sin(i->phi()),
				     z);
	   eldirection->SetMarkerSize(1);
	}
	eldirection->SetMarkerColor(kYellow);
	tList->AddElement(eldirection);
     }
}

void
FWElectronDetailView::setBackgroundColor(Color_t col)
{
   FWColorManager::setColorSetViewer(m_viewer, col);
}

REGISTER_FWDETAILVIEW(FWElectronDetailView);
