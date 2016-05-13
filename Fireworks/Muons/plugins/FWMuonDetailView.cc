
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWMuonDetailView
//

// ROOT includes
#include "TLatex.h"
#include "TEveCalo.h"
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TCanvas.h"
#include "TEveCaloLegoOverlay.h"
#include "TRootEmbeddedCanvas.h"

// Fireworks includes
#include "Fireworks/Muons/plugins/FWMuonDetailView.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWGLEventHandler.h"

#include "DataFormats/MuonReco/interface/Muon.h"
//
// constructors and destructor
//
FWMuonDetailView::FWMuonDetailView():
   m_data(0),
   m_builder(0)
{ 
}

FWMuonDetailView::~FWMuonDetailView()
{
   m_eveViewer->GetGLViewer()->DeleteOverlayElements(TGLOverlayElement::kUser);
   if (m_data) m_data->DecDenyDestroy();
   delete m_builder;
}

//
// member functions
//
void FWMuonDetailView::build(const FWModelId &id, const reco::Muon* iMuon)
{
   if(0==iMuon) return;

   double eta = iMuon->eta();
   double phi = iMuon->phi();
   if ( iMuon->isEnergyValid() )
   {
      eta = iMuon->calEnergy().ecal_position.eta();
      phi = iMuon->calEnergy().ecal_position.phi();
   }

   m_builder = new FWECALDetailViewBuilder(id.item()->getEvent(), id.item()->getGeom(),
                                           eta, phi, 10);
 
   m_builder->showSuperClusters();
   if ( iMuon->isEnergyValid() ) {
      std::vector<DetId> ids;
      ids.push_back(iMuon->calEnergy().ecal_id);
      m_builder->setColor(kYellow,ids);
   }

   TEveCaloLego* lego = m_builder->build();
   m_data = lego->GetData();
   m_data->IncDenyDestroy();
   m_eveScene->AddElement(lego);
   addSceneInfo(iMuon, m_eveScene);

   // draw axis at the window corners
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   overlay->SetShowPlane(kFALSE);
   overlay->SetShowPerspective(kFALSE);
   overlay->SetCaloLego(lego);
   overlay->SetShowScales(0); // temporary
   viewerGL()->AddOverlayElement(overlay);

   // set event handler and flip camera to top view at beginning
   viewerGL()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   FWGLEventHandler* eh =
      new FWGLEventHandler((TGWindow*)viewerGL()->GetGLWidget(), (TObject*)viewerGL(), lego);
   viewerGL()->SetEventHandler(eh);

   viewerGL()->ResetCamerasAfterNextUpdate();
   viewerGL()->UpdateScene(kFALSE);
   gEve->Redraw3D();

   setTextInfo(id, iMuon);
}

void
FWMuonDetailView::setTextInfo(const FWModelId& id, const reco::Muon *muon)
{
   m_infoCanvas->cd();
   double y0 = 0.95;
   double x0 = 0.02; 

   TLatex* latex = new TLatex(x0, y0, "");
   const double textsize(0.05);
   latex->SetTextSize(2*textsize);

   float_t x = x0;
   float   y = y0;
   float fontsize = latex->GetTextSize()*0.6;

   latex->DrawLatex(x, y, id.item()->modelName(id.index()).c_str() );
   y -= fontsize;
   latex->SetTextSize(textsize);
   fontsize = latex->GetTextSize()*0.6;

   latex->DrawLatex(x, y, Form(" p_{T} = %.1f GeV, #eta = %0.2f, #varphi = %0.2f",
			       muon->pt(), muon->eta(), muon->phi()) );
   y -= fontsize;
   // summary
   if (muon->charge() > 0)
      latex->DrawLatex(x, y, " charge = +1");
   else
      latex->DrawLatex(x, y, " charge = -1");
   y -= fontsize;

   if (muon->isEnergyValid() )
   {
      // delta phi/eta in
      latex->DrawLatex(x, y, "ECAL energy in:");
      y -= fontsize;
      latex->DrawLatex(x, y,  Form(" crossed crystalls = %.3f",
                                   muon->calEnergy().em) );
      y -= fontsize;
      latex->DrawLatex(x, y,  Form(" 3x3 crystall shape = %.3f",
                                   muon->calEnergy().emS9) );
      y -= fontsize;
      latex->DrawLatex(x, y,  Form(" 5x5 crystall shape = %.3f",
                                   muon->calEnergy().emS25) );
   }
   m_builder->makeLegend(0.02, y);
}

void
FWMuonDetailView::addSceneInfo(const reco::Muon *i, TEveElementList* tList)
{
   // muon direction at vertex

   bool barrel = fabs(i->eta())<1.5;
   if ( i->isEnergyValid() ) barrel = fabs(i->calEnergy().ecal_position.eta()) < 1.5;

   Double_t x(0), y(0);
   Double_t delta(0.01);
   TEveStraightLineSet *direction = new TEveStraightLineSet("Direction at vertex");
   direction->SetPickable(kTRUE);
   direction->SetTitle("Muon direction at vertex");
   direction->SetPickable(kTRUE);
   if (barrel) {
     x = i->eta();
     y = i->phi();
   }else{
     x = 310*fabs(tan(i->theta()))*cos(i->phi());
     y = 310*fabs(tan(i->theta()))*sin(i->phi());
   }
   direction->AddLine(x-delta,y-delta,0,x+delta,y+delta,0);
   direction->AddLine(x-delta,y+delta,0,x+delta,y-delta,0);
   direction->SetLineColor(kYellow);
   direction->SetDepthTest(kFALSE);
   tList->AddElement(direction);

   if (! i->isEnergyValid() ) return;

   // ecal position
   TEveStraightLineSet *ecalposition = new TEveStraightLineSet("ecal position");
   ecalposition->SetPickable(kTRUE);
   ecalposition->SetTitle("Muon position at ECAL surface");
   if ( barrel ) {
      x = i->calEnergy().ecal_position.eta();
      y = i->calEnergy().ecal_position.phi();
   } else {
      x = i->calEnergy().ecal_position.x();
      y = i->calEnergy().ecal_position.y();
   }
   ecalposition->AddLine(x-delta,y,0,x+delta,y,0);
   ecalposition->AddLine(x,y-delta,0,x,y+delta,0);
   ecalposition->AddLine(x,y,0-delta,x,y,0+delta);
   ecalposition->SetLineColor(kRed);
   ecalposition->SetLineWidth(2);
   ecalposition->SetDepthTest(kFALSE);
   tList->AddElement(ecalposition);

   // hcal position
   TEveStraightLineSet *hcalposition = new TEveStraightLineSet("hcal position");
   hcalposition->SetPickable(kTRUE);
   hcalposition->SetTitle("Muon position at HCAL surface");
   if ( barrel ) {
      x = i->calEnergy().hcal_position.eta();
      y = i->calEnergy().hcal_position.phi();
   } else {
      x = i->calEnergy().hcal_position.x();
      y = i->calEnergy().hcal_position.y();
   }
   hcalposition->AddLine(x-delta,y,0,x+delta,y,0);
   hcalposition->AddLine(x,y-delta,0,x,y+delta,0);
   hcalposition->AddLine(x,y,0-delta,x,y,0+delta);
   hcalposition->SetLineColor(kBlue);
   hcalposition->SetLineWidth(2);
   hcalposition->SetDepthTest(kFALSE);
   tList->AddElement(hcalposition);

   // draw a line connecting the two positions
   TEveStraightLineSet *lines = new TEveStraightLineSet("Muon trajectory in ECAL","Muon trajectory in ECAL");
   lines->SetPickable(kTRUE);
   if ( barrel ) {
      lines->AddLine(i->calEnergy().ecal_position.eta(),
		     i->calEnergy().ecal_position.phi(),
		     0,
		     i->calEnergy().hcal_position.eta(),
		     i->calEnergy().hcal_position.phi(),
		     0);
   } else {
      lines->AddLine(i->calEnergy().ecal_position.x(),
		     i->calEnergy().ecal_position.y(),
		     0,
		     i->calEnergy().hcal_position.x(),
		     i->calEnergy().hcal_position.y(),
		     0);
   }
   lines->SetLineColor(kRed);
   lines->SetDepthTest(kFALSE);
   tList->AddElement(lines);
}


REGISTER_FWDETAILVIEW(FWMuonDetailView,Muon,ecalRecHit);
REGISTER_FWDETAILVIEW(FWMuonDetailView,Muon,reducedEcalRecHitsEB);
