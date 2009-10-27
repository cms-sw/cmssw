//
// Package:     Calo
// Class  :     FWPhotonDetailView
// $Id: FWPhotonDetailView.cc,v 1.19 2009/10/12 17:54:05 amraktad Exp $

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
   getEveWindow()->DestroyWindow();
}

//
// member functions
//
void FWPhotonDetailView::build (const FWModelId &id, const reco::Photon* iPhoton, TEveWindowSlot* base)
{
   if(0==iPhoton) return;

   TEveScene*  scene(0);
   TEveViewer* viewer(0);
   TCanvas*    canvas(0);
   TEveWindow* eveWindow = FWDetailViewBase::makePackViewer(base, canvas, viewer, scene);
   setEveWindow(eveWindow);
   m_viewer = viewer->GetGLViewer();
   makeLegend(iPhoton, id, canvas);

   // build ECAL objects
   FWECALDetailViewBuilder builder(id.item()->getEvent(), id.item()->getGeom(),
   iPhoton->caloPosition().eta(), iPhoton->caloPosition().phi(), 25);
   builder.showSuperClusters(kGreen+2, kGreen+4);
   if ( iPhoton->superCluster().isAvailable() )
   builder.showSuperCluster(*(iPhoton->superCluster()), kYellow);
   TEveCaloLego* lego = builder.build();
   m_data = lego->GetData();
   scene->AddElement(lego);
   
   // add Photon specific details
   addInfo(iPhoton, scene);
   
   // draw axis at the window corners
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

   m_viewer->RequestDraw(TGLRnrCtx::kLODHigh);
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
