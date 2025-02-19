// -*- C++ -*-
//
// Package:     Electrons
// Class  :     FWElectronDetailView
// $Id: FWElectronDetailView.cc,v 1.59 2011/02/28 10:32:01 amraktad Exp $
//

// ROOT includes
#include "TLatex.h"
#include "TEveCalo.h"
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGLOverlay.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TEveCaloLegoOverlay.h"
#include "TRootEmbeddedCanvas.h"

// Fireworks includes
#include "Fireworks/Electrons/plugins/FWElectronDetailView.h"
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGLEventHandler.h"

// CMSSW includes
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


//
// constructors and destructor
//
FWElectronDetailView::FWElectronDetailView() :
   m_data(0),
   m_builder(0),
   m_legend(0)
{
}

FWElectronDetailView::~FWElectronDetailView()
{  
   m_eveViewer->GetGLViewer()->DeleteOverlayElements(TGLOverlayElement::kUser);

   delete m_builder;
   if (m_data) m_data->DecDenyDestroy();
}

//
// member functions
//
void
FWElectronDetailView::build( const FWModelId &id, const reco::GsfElectron* iElectron )
{
   if( !iElectron ) return;
   // If SuperCluster reference is not stored,
   // take eta and phi of a Candidate
   double eta = 0;
   double phi = 0;
   if( iElectron->superCluster().isAvailable() ) {
      eta = iElectron->caloPosition().eta();
      phi = iElectron->caloPosition().phi();
   }
   else 
   {
      eta = iElectron->eta();
      phi = iElectron->phi();
   }

   // build ECAL objects
   m_builder = new FWECALDetailViewBuilder( id.item()->getEvent(), id.item()->getGeom(),
					    eta, phi, 25);
 
   m_builder->showSuperClusters();
   if( iElectron->superCluster().isAvailable() )
      m_builder->showSuperCluster( *(iElectron->superCluster() ), kYellow);
   TEveCaloLego* lego = m_builder->build();
   m_data = lego->GetData();
   m_eveScene->AddElement( lego );

   m_legend = new TLegend(0.01, 0.01, 0.99, 0.99, 0, "NDC");
   m_legend->SetTextSize(0.075);
   m_legend->SetBorderSize(0);
   m_legend->SetMargin(0.15);
   m_legend->SetEntrySeparation(0.05);

   // add Electron specific details
   if( iElectron->superCluster().isAvailable() ) {
      addTrackPointsInCaloData( iElectron, lego );
      drawCrossHair( iElectron, lego, m_eveScene );
      addSceneInfo( iElectron, m_eveScene );
   }
   
   // draw axis at the window corners
   if (1)
   {
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   overlay->SetShowPlane( kFALSE );
   overlay->SetShowPerspective( kFALSE );
   overlay->SetCaloLego( lego );
   overlay->SetShowScales( 1 ); // temporary
   viewerGL()->AddOverlayElement( overlay );
   }
   // set event handler and flip camera to top view at beginning
   viewerGL()->SetCurrentCamera( TGLViewer::kCameraOrthoXOY );
   FWGLEventHandler* eh =
      new FWGLEventHandler( (TGWindow*)viewerGL()->GetGLWidget(), (TObject*)viewerGL(), lego );
   viewerGL()->SetEventHandler( eh );
   viewerGL()->ResetCamerasAfterNextUpdate();
   viewerGL()->UpdateScene(kFALSE);
   gEve->Redraw3D();

   setTextInfo( id, iElectron );
}

double
FWElectronDetailView::deltaEtaSuperClusterTrackAtVtx( const reco::GsfElectron &electron )
{
   return electron.deltaEtaSuperClusterTrackAtVtx();
}

double
FWElectronDetailView::deltaPhiSuperClusterTrackAtVtx( const reco::GsfElectron &electron )
{
   return electron.deltaPhiSuperClusterTrackAtVtx();
}

void
FWElectronDetailView::setTextInfo( const FWModelId& id, const reco::GsfElectron *electron )
{
   m_infoCanvas->cd();

   float_t x  = 0.02;
   float_t x2 = 0.52;
   float   y  = 0.95;

   TLatex* latex = new TLatex( x, y, "" );
   const double textsize( 0.05 );
   latex->SetTextSize( 2*textsize );

   latex->DrawLatex( x, y, id.item()->modelName( id.index() ).c_str() );
   y -= latex->GetTextSize()*0.6;

   latex->SetTextSize( textsize );
   float lineH = latex->GetTextSize()*0.6;

   latex->DrawLatex( x, y, Form( " E_{T} = %.1f GeV, #eta = %0.2f, #varphi = %0.2f",
				 electron->et(), electron->eta(), electron->phi()) );
   y -= lineH;
   // summary
   if( electron->charge() > 0 )
      latex->DrawLatex( x, y, " charge = +1" );
   else
      latex->DrawLatex( x, y, " charge = -1" );
   y -= lineH;

   if( electron->superCluster().isAvailable() ) {     
      // delta phi/eta in
      latex->DrawLatex( x, y, "SuperCluster vs inner state extrapolation" );
      y -= lineH;
      latex->DrawLatex(  x, y, TString::Format(" #Delta#eta_{in} = %.3f",   electron->deltaEtaSuperClusterTrackAtVtx()) );
      latex->DrawLatex( x2, y, TString::Format("#Delta#varphi_{in} = %.3f", electron->deltaPhiSuperClusterTrackAtVtx()) );
      y -= lineH;

      // delta phi/eta out
      latex->DrawLatex( x, y, "SeedCluster vs outer state extrapolation" );
      y -= lineH;

      latex->DrawLatex(  x, y, TString::Format(" #Delta#eta_{out} = %.3f",    electron->deltaEtaSeedClusterTrackAtCalo()) );
      latex->DrawLatex( x2, y, TString::Format(" #Delta#varphi_{out} = %.3f", electron->deltaPhiSeedClusterTrackAtCalo()) );
      y -= 2*lineH;
   } else {
     latex->DrawLatex( x, y, "Ref to SuperCluster is not available" );
   }

   latex->DrawLatex(x, y, TString::Format(" Tracker driven seed: %s", electron->trackerDrivenSeed() ? "YES" : "NO"));
   y -= lineH;
   latex->DrawLatex(x, y, TString::Format(" ECAL driven seed: %s",    electron->ecalDrivenSeed() ? "YES" : "NO"));
   y -= lineH;

   y = m_builder->makeLegend( 0.02, y );
   y -= lineH;

   m_legend->SetY2(y);
   m_legend->Draw();
   m_legend = 0; // Deleted together with TPad.
}

void
FWElectronDetailView::drawCrossHair (const reco::GsfElectron* i, TEveCaloLego *lego, TEveElementList* tList)
{
   unsigned int subdetId( 0 );
   
   if( !i->superCluster()->seed()->hitsAndFractions().empty() )
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

      m_legend->AddEntry(trackpositionAtCalo, "From outermost state", "l");
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

      m_legend->AddEntry(pinposition, "From innermost state", "l");
   }
}

Bool_t
FWElectronDetailView::checkRange( Double_t &em, Double_t& eM, Double_t &pm, Double_t& pM,
				  Double_t eta, Double_t phi )
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
FWElectronDetailView::addTrackPointsInCaloData( const reco::GsfElectron *i, TEveCaloLego* lego )
{
   unsigned int subdetId(0);

   if ( !i->superCluster()->seed()->hitsAndFractions().empty() )
      subdetId = i->superCluster()->seed()->hitsAndFractions().front().first.subdetId();

   TEveCaloDataVec* data = (TEveCaloDataVec*)lego->GetData();
   Double_t em, eM, pm, pM;
   data->GetEtaLimits(em, eM);
   data->GetPhiLimits(pm, pM);
   data->IncDenyDestroy();
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
FWElectronDetailView::addSceneInfo(const reco::GsfElectron *i, TEveElementList* tList)
{
   unsigned int subdetId(0);

   if ( !i->superCluster()->seed()->hitsAndFractions().empty() )
      subdetId = i->superCluster()->seed()->hitsAndFractions().front().first.subdetId();

   // centroids
   Double_t x(0), y(0), z(0);
   Double_t delta(0.02);
   if (subdetId == EcalEndcap) delta = 2.5;
   TEveStraightLineSet *scposition = new TEveStraightLineSet("sc position");
   scposition->SetPickable(kTRUE);
   scposition->SetTitle("Super cluster centroid");
   if (subdetId == EcalBarrel) {
      x = i->caloPosition().eta();
      y = i->caloPosition().phi();
   } else if (subdetId == EcalEndcap) {
      x = i->caloPosition().x();
      y = i->caloPosition().y();
   }
   scposition->AddLine(x-delta,y,z,x+delta,y,z);
   scposition->AddLine(x,y-delta,z,x,y+delta,z);
   scposition->AddLine(x,y,z-delta,x,y,z+delta);
   scposition->SetLineColor(kBlue);
   scposition->SetLineWidth(2);
   scposition->SetDepthTest(kFALSE);
   tList->AddElement(scposition);

   scposition->SetMarkerColor(kBlue);
   scposition->SetMarkerStyle(2);
   m_legend->AddEntry(scposition, "Super cluster centroid", "p");

   // seed position
   TEveStraightLineSet *seedposition = new TEveStraightLineSet("seed position");
   seedposition->SetTitle("Seed cluster centroid");
   seedposition->SetPickable(kTRUE);
   if (subdetId == EcalBarrel) {
      x  = i->superCluster()->seed()->position().eta();
      y  = i->superCluster()->seed()->position().phi();
   } else if (subdetId == EcalEndcap) {
      x  = i->superCluster()->seed()->position().x();
      y  = i->superCluster()->seed()->position().y();
   }
   seedposition->AddLine(x-delta,y-delta,z,x+delta,y+delta,z);
   seedposition->AddLine(x-delta,y+delta,z,x+delta,y-delta,z);
   seedposition->SetLineColor(kRed);
   seedposition->SetLineWidth(2);
   seedposition->SetDepthTest(kFALSE);
   tList->AddElement(seedposition);

   seedposition->SetMarkerColor(kRed);
   seedposition->SetMarkerStyle(5);
   m_legend->AddEntry(seedposition, "Seed cluster centroid", "p");

   // electron direction (show it if it's within
   // the area of interest)
   if ( fabs(i->phi()-i->caloPosition().phi())< 25*0.0172 &&
        fabs(i->eta()-i->caloPosition().eta())< 25*0.0172 )
   {
      TEveStraightLineSet *eldirection = new TEveStraightLineSet("seed position");
      eldirection->SetTitle("Electron direction at vertex");
      eldirection->SetPickable(kTRUE);
      if (subdetId == EcalBarrel) {
         x = i->eta();
         y = i->phi();
      }else{
         x = 310*fabs(tan(i->theta()))*cos(i->phi());
         y = 310*fabs(tan(i->theta()))*sin(i->phi());
      }
      eldirection->AddLine(x-delta,y-delta,z,x+delta,y+delta,z);
      eldirection->AddLine(x-delta,y+delta,z,x+delta,y-delta,z);
      eldirection->SetLineColor(kGreen);
      eldirection->SetDepthTest(kFALSE);
      tList->AddElement(eldirection);

      eldirection->SetMarkerColor(kGreen);
      eldirection->SetMarkerStyle(5);
      m_legend->AddEntry(eldirection, "Direction at vertex", "p");
   }
}

REGISTER_FWDETAILVIEW(FWElectronDetailView,Electron);
