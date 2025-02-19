//
// Package:     Electrons
// Class  :     FWPhotonDetailView
// $Id: FWPhotonDetailView.cc,v 1.30 2011/11/18 02:57:08 amraktad Exp $

#include "TLatex.h"
#include "TEveCalo.h"
#include "TEvePointSet.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TCanvas.h"
#include "TEveCaloLegoOverlay.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "Fireworks/Electrons/plugins/FWPhotonDetailView.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGLEventHandler.h"

//
// constructors and destructor
//
FWPhotonDetailView::FWPhotonDetailView():
m_data(0),
m_builder(0)
{
}

FWPhotonDetailView::~FWPhotonDetailView()
{
   m_eveViewer->GetGLViewer()->DeleteOverlayElements(TGLOverlayElement::kUser);

   if (m_data) m_data->DecDenyDestroy();
   delete m_builder;
}

//
// member functions
//
void FWPhotonDetailView::build (const FWModelId &id, const reco::Photon* iPhoton)
{
   if(!iPhoton) return;

   // build ECAL objects
   m_builder = new FWECALDetailViewBuilder(id.item()->getEvent(), id.item()->getGeom(),
                                   iPhoton->caloPosition().eta(), iPhoton->caloPosition().phi(), 25);
   m_builder->showSuperClusters();

   if ( iPhoton->superCluster().isAvailable() )
      m_builder->showSuperCluster(*(iPhoton->superCluster()), kYellow);

   TEveCaloLego* lego = m_builder->build();
   m_data = lego->GetData();
   m_data->IncDenyDestroy();
   m_eveScene->AddElement(lego);

   // add Photon specific details
   if( iPhoton->superCluster().isAvailable() ) 
      addSceneInfo(iPhoton, m_eveScene);

   // draw axis at the window corners
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   overlay->SetShowPlane(kFALSE);
   overlay->SetShowPerspective(kFALSE);
   overlay->SetCaloLego(lego);
   overlay->SetShowScales(1); // temporary
   viewerGL()->AddOverlayElement(overlay);

   // set event handler and flip camera to top view at beginning
   viewerGL()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   FWGLEventHandler* eh =
      new FWGLEventHandler((TGWindow*)viewerGL()->GetGLWidget(), (TObject*)viewerGL(), lego);
   viewerGL()->SetEventHandler(eh);
   viewerGL()->UpdateScene();
   viewerGL()->CurrentCamera().Reset();

   viewerGL()->RequestDraw(TGLRnrCtx::kLODHigh);

   setTextInfo(id, iPhoton);
}

//______________________________________________________________________________

void
FWPhotonDetailView::setTextInfo(const FWModelId& id, const reco::Photon *photon)
{
   m_infoCanvas->cd();
   float_t x = 0.02;
   float y = 0.97;
   TLatex* latex = new TLatex(x, y, "");
   const double textsize(0.05);
   latex->SetTextSize(2*textsize);

   float h = latex->GetTextSize()*0.6;
   latex->DrawLatex(x, y, id.item()->modelName(id.index()).c_str() );
   y -= h;

   latex->DrawLatex(x, y, Form(" E_{T} = %.1f GeV, #eta = %0.2f, #varphi = %0.2f",
                               photon->et(), photon->eta(), photon->phi()) );
   y -= h;
   m_builder->makeLegend(x, y);
}

//______________________________________________________________________________

void
FWPhotonDetailView::addSceneInfo(const reco::Photon *i, TEveElementList* tList)
{
   unsigned int subdetId(0);
   if ( !i->superCluster()->seed()->hitsAndFractions().empty() )
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

REGISTER_FWDETAILVIEW(FWPhotonDetailView,Photon);
