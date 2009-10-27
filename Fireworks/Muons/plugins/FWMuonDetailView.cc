// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWMuonDetailView
// $Id: FWMuonDetailView.cc,v 1.15 2009/10/12 17:54:06 amraktad Exp $
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

// Fireworks includes
#include "Fireworks/Muons/plugins/FWMuonDetailView.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWColorManager.h"
//
// constructors and destructor
//
FWMuonDetailView::FWMuonDetailView():
   m_viewer(0),
   m_data(0)
{ 
   getEveWindow()->DestroyWindow();
}

FWMuonDetailView::~FWMuonDetailView()
{
   getEveWindow()->DestroyWindow();
   m_data->DecDenyDestroy();
}

//
// member functions
//
void FWMuonDetailView::build(const FWModelId &id, const reco::Muon* iMuon, TEveWindowSlot* base)
{

   if(0==iMuon) return;

   TEveScene*  scene(0);
   TEveViewer* viewer(0);
   TCanvas*    canvas(0);
   TEveWindow* eveWindow = FWDetailViewBase::makePackViewer(base, canvas, viewer, scene);
   setEveWindow(eveWindow);
   m_viewer = viewer->GetGLViewer();
   makeLegend(iMuon, id, canvas);

   double eta = iMuon->eta();
   double phi = iMuon->phi();
   if ( iMuon->isEnergyValid() )
   {
      eta = iMuon->calEnergy().ecal_position.eta();
      phi = iMuon->calEnergy().ecal_position.phi();
   }

   FWECALDetailViewBuilder builder(id.item()->getEvent(), id.item()->getGeom(),
                                   eta, phi, 10);
   builder.showSuperClusters(kGreen+2, kGreen+4);
   if ( iMuon->isEnergyValid() ) {
      std::vector<DetId> ids;
      ids.push_back(iMuon->calEnergy().ecal_id);
      builder.setColor(kYellow,ids);
   }

   TEveCaloLego* lego = builder.build();
   m_data = lego->GetData();
   scene->AddElement(lego);
   addInfo(iMuon, scene);

   // draw axis at the window corners
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   overlay->SetShowPlane(kFALSE);
   overlay->SetShowPerspective(kFALSE);
   overlay->SetCaloLego(lego);
   overlay->SetShowScales(0); // temporary
   m_viewer->AddOverlayElement(overlay);

   // set event handler and flip camera to top view at beginning
   m_viewer->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TEveLegoEventHandler* eh =
      new TEveLegoEventHandler((TGWindow*)m_viewer->GetGLWidget(), (TObject*)m_viewer, lego);
   m_viewer->SetEventHandler(eh);
   m_viewer->CurrentCamera().Reset();

   viewer->GetGLViewer()->RequestDraw(TGLRnrCtx::kLODHigh);
   gEve->Redraw3D();
}

void
FWMuonDetailView::makeLegend(const reco::Muon *muon,
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

   latex->DrawLatex(x, y, Form(" p_{T} = %.1f GeV, #eta = %0.2f, #varphi = %0.2f",
			       muon->pt(), muon->eta(), muon->phi()) );
   y -= fontsize;
   // summary
   if (muon->charge() > 0)
     latex->DrawLatex(x, y, " charge = +1");
   else
     latex->DrawLatex(x, y, " charge = -1");
   y -= fontsize;

   if (! muon->isEnergyValid() ) return;
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

void
FWMuonDetailView::addInfo(const reco::Muon *i, TEveElementList* tList)
{
   // muon direction at vertex

   bool barrel = fabs(i->eta())<1.5;
   if ( i->isEnergyValid() ) barrel = fabs(i->calEnergy().ecal_position.eta()) < 1.5;

   TEvePointSet *direction = new TEvePointSet("muon direction");
   direction->SetTitle("Muon direction at vertex");
   direction->SetPickable(kTRUE);
   direction->SetMarkerStyle(2);
   if (barrel) {
      direction->SetNextPoint(i->eta(), i->phi(), 0);
      direction->SetMarkerSize(0.01);
   }else{
      direction->SetNextPoint(310*fabs(tan(i->theta()))*cos(i->phi()),
			      310*fabs(tan(i->theta()))*sin(i->phi()),
			      0);
      direction->SetMarkerSize(1);
   }
   direction->SetMarkerColor(kYellow);
   tList->AddElement(direction);

   if (! i->isEnergyValid() ) return;

   // ecal position
   Double_t x(0), y(0);
   TEvePointSet *ecalposition = new TEvePointSet("ecal position");
   ecalposition->SetPickable(kTRUE);
   ecalposition->SetTitle("Track position at ECAL surface");
   if ( barrel ) {
      x = i->calEnergy().ecal_position.eta();
      y = i->calEnergy().ecal_position.phi();
   } else {
      x = i->calEnergy().ecal_position.x();
      y = i->calEnergy().ecal_position.y();
   }
   ecalposition->SetNextPoint(x,y,0);
   ecalposition->SetMarkerSize(1);
   ecalposition->SetMarkerStyle(4);
   ecalposition->SetMarkerColor(kRed);
   tList->AddElement(ecalposition);

   // hcal position
   TEvePointSet *hcalposition = new TEvePointSet("hcal position");
   hcalposition->SetTitle("Track position at HCAL surface");
   hcalposition->SetPickable(kTRUE);
   if ( barrel ) {
      x = i->calEnergy().hcal_position.eta();
      y = i->calEnergy().hcal_position.phi();
      hcalposition->SetMarkerSize(0.01);
   } else {
      x = i->calEnergy().hcal_position.x();
      y = i->calEnergy().hcal_position.y();
      hcalposition->SetMarkerSize(1);
   }
   hcalposition->SetNextPoint(x, y, 0);
   hcalposition->SetMarkerStyle(2);
   hcalposition->SetMarkerColor(kRed);
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
   tList->AddElement(lines);
}


void
FWMuonDetailView::setBackgroundColor(Color_t col)
{
   FWColorManager::setColorSetViewer(m_viewer, col);
}

REGISTER_FWDETAILVIEW(FWMuonDetailView);
