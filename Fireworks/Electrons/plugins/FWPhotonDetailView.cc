//
// Package:     Calo
// Class  :     FWPhotonDetailView
// $Id: FWPhotonDetailView.cc,v 1.16 2009/09/24 20:19:07 amraktad Exp $

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
#include "TEveLegoEventHandler.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "Fireworks/Electrons/plugins/FWPhotonDetailView.h"

#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWColorManager.h"
//
// constructors and destructor
//
FWPhotonDetailView::FWPhotonDetailView():
m_viewer(0)
{
}

FWPhotonDetailView::~FWPhotonDetailView()
{
}

//
// member functions
//
void FWPhotonDetailView::build (const FWModelId &id, const reco::Photon* iPhoton, TEveWindowSlot* base)
{
   if(0==iPhoton) return;

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
   ew->SetElementName("Photon based view");
   FWDetailViewBase::setEveWindow(ew);
   
   // build ECAL objects
   FWECALDetailViewBuilder builder(id.item()->getEvent(), id.item()->getGeom(),
				   iPhoton->caloPosition().eta(), iPhoton->caloPosition().phi(), 25);
   builder.showSuperClusters(kGreen+2, kGreen+4);
   if ( iPhoton->superCluster().isAvailable() )
     builder.showSuperCluster(*(iPhoton->superCluster()), kYellow);
   TEveCaloLego* lego = builder.build();
   scene->AddElement(lego);
   
   // add Photon specific details
   addInfo(iPhoton, scene);

   TRootEmbeddedCanvas* ec = new TRootEmbeddedCanvas("Embeddedcanvas", ediFrame, 100, 100, 0);
   ediFrame->AddFrame(ec, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   ediFrame->MapSubwindows();
   ediFrame->Layout();
   ec->GetCanvas()->SetBorderMode(0);
   makeLegend(iPhoton, id, ec->GetCanvas());
   
   // draw axis at the window corners
   // std::cout << "TEveViewer: " << viewer << std::endl;
   TGLViewer* glv =  viewer->GetGLViewer();
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   overlay->SetShowPlane(kFALSE);
   overlay->SetShowPerspective(kFALSE);
   overlay->SetCaloLego(lego);
   overlay->SetShowScales(0); // temporary
   glv->AddOverlayElement(overlay);

   // set event handler and flip camera to top view at beginning
   glv->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TEveLegoEventHandler* eh = 
      new TEveLegoEventHandler((TGWindow*)glv->GetGLWidget(), (TObject*)glv, lego);
   glv->SetEventHandler(eh);
   glv->UpdateScene();
   glv->CurrentCamera().Reset();

   scene->Repaint(true);
   viewer->GetGLViewer()->RequestDraw(TGLRnrCtx::kLODHigh);
   gEve->Redraw3D();
   
   ////////////////////////////////////////////////////////////////////////
   //                              Sub-view 2
   ///////////////////////////////////////////////////////////////////////
}

void
FWPhotonDetailView::makeLegend(const reco::Photon *photon,
			       const FWModelId& id,
			       TCanvas* textCanvas)
{
   textCanvas->cd();

   TLatex* latex = new TLatex(0.02, 0.970, "");
   const double textsize(0.05);
   latex->SetTextSize(2*textsize);

   float_t x = 0.02;
   float   y = 0.95;
   float fontsize = latex->GetTextSize()*0.6;
   
   latex->DrawLatex(x, y, id.item()->modelName(id.index()).c_str() );
   y -= fontsize;
   latex->SetTextSize(textsize);
   fontsize = latex->GetTextSize()*0.6;

   latex->DrawLatex(x, y, Form(" E_{T} = %.1f GeV, #eta = %0.2f, #varphi = %0.2f", 
			       photon->et(), photon->eta(), photon->phi()) );
}

void
FWPhotonDetailView::addInfo(const reco::Photon *i, TEveElementList* tList)
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
}

void
FWPhotonDetailView::setBackgroundColor(Color_t col)
{
   FWColorManager::setColorSetViewer(m_viewer, col);
}

REGISTER_FWDETAILVIEW(FWPhotonDetailView);
