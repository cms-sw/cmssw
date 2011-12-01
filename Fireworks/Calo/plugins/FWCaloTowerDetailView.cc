#include "TLatex.h"
#include "TEveCalo.h"
#include "TEveScene.h"
#include "TGLViewer.h"
#include "TCanvas.h"
#include "TEveCaloLegoOverlay.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"

#include "Fireworks/Calo/plugins/FWCaloTowerDetailView.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGLEventHandler.h"

//
// constructors and destructor
//
FWCaloTowerDetailView::FWCaloTowerDetailView():
  m_data(0),
  m_builder(0)
{ 
}

FWCaloTowerDetailView::~FWCaloTowerDetailView()
{
}

//
// member functions
//
void FWCaloTowerDetailView::build(const FWModelId &id, const CaloTower* iTower)
{
   if(!iTower) return;

   // build ECAL objects
   m_builder = new FWECALDetailViewBuilder(id.item()->getEvent(), id.item()->getGeom(),
					   iTower->eta(), iTower->phi(), 25);
   m_builder->showSuperClusters();
   
   TEveCaloLego* lego = m_builder->build();
   m_data = lego->GetData();
   m_data->IncDenyDestroy();
   m_eveScene->AddElement(lego);

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

   setTextInfo(id, iTower);

}

void
FWCaloTowerDetailView::setTextInfo(const FWModelId& id, const CaloTower* tower)
{
   m_infoCanvas->cd();
   float_t x = 0.02;
   float y = 0.97;
   TLatex* latex = new TLatex(x, y, "");
   const double textsize(0.05);
   latex->SetTextSize(textsize);

   float h = latex->GetTextSize()*0.6;
   latex->DrawLatex(x, y, "ECAL hit detail view centered on tower:" );
   y -= h;
   latex->DrawLatex(x, y, Form(" %s",id.item()->modelName(id.index()).c_str()) );
   y -= h;
   latex->DrawLatex(x, y, Form(" E_{T}(em) = %.1f GeV, E_{T}(had) = %.1f GeV",
                               tower->emEt(), tower->hadEt()) );
   y -= h;
   latex->DrawLatex(x, y, Form(" #eta = %0.2f, #varphi = %0.2f",
                               tower->eta(), tower->phi()) );
   y -= h;
   m_builder->makeLegend(x, y);
}

REGISTER_FWDETAILVIEW(FWCaloTowerDetailView, Tower);
