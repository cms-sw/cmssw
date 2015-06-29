// ROOT includes
#include "TEveScene.h"
#include "TEveManager.h"
#include "TEveStraightLineSet.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveTrans.h"
#include "TEveText.h"
#include "TEveGeoShape.h"
#include "TGLViewer.h"
#include "TGLScenePad.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TLegend.h"

#include "TH2.h"

#include "TAxis.h"
#include "TGSlider.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGLCameraOverlay.h"

// boost includes
#include "boost/bind.hpp"

#include "Fireworks/ParticleFlow/plugins/FWPFCandidateDetailView.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"

#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/FWIntValueListener.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/ParticleFlow/interface/FWPFMaths.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"


FWPFCandidateDetailView::FWPFCandidateDetailView ():
   m_range(1),
   m_candidate(0),
   m_legend(0),
   m_slider(0),
   m_sliderListener(),
   m_eventList(0),
   m_plotEt(true),
   m_rnrHcal(true)
{}

FWPFCandidateDetailView::~FWPFCandidateDetailView ()
{
}


float FWPFCandidateDetailView::eta()
{
  return m_candidate->eta();
}

float FWPFCandidateDetailView::phi()
{
   return m_candidate->phi();
}

bool FWPFCandidateDetailView::isPntInRng(float x, float y)
{
   float dx = m_candidate->eta() - x;
   float dy = m_candidate->phi() - y;
   float sd = TMath::Sqrt(dx*dx + dy*dy);
   return sd < m_range;
}

//______________________________________________________________________________

void
FWPFCandidateDetailView::makeLegend()
{
   m_legend = new TLegend( 0.01, 0.01, 0.99, 0.99, 0, "NDC" );
   m_legend->SetFillColor(kWhite);
   m_legend->SetTextSize( 0.07 );
   m_legend->SetBorderSize( 0 );
   m_legend->SetMargin( 0.15 );
   m_legend->SetEntrySeparation( 0.01 );
}

//______________________________________________________________________________

void
FWPFCandidateDetailView::rangeChanged(int x)
{
   static float kSliderRangeFactor = 0.2;

   m_range = x * kSliderRangeFactor;

   if (m_eventList) buildGLEventScene();


   gEve->Redraw3D();
}

//______________________________________________________________________________

void
FWPFCandidateDetailView::setTextInfo(const FWModelId &id, const reco::PFCandidate* track)
{
   m_infoCanvas->cd();

   float_t x  = 0.02;
   float   y  = 0.95;

   TLatex* latex = new TLatex( x, y, "" );
   const double textsize( 0.07 );
   latex->SetTextSize( textsize );

   latex->DrawLatex( x, y, id.item()->modelName( id.index()).c_str());
   y -= latex->GetTextSize()*0.6;

   latex->SetTextSize( textsize );
   float lineH = latex->GetTextSize()*0.6;

   latex->DrawLatex( x, y, Form( " P_{T} = %.1f GeV, #eta = %0.2f, #varphi = %0.2f",
				 track->pt(), track->eta(), track->phi()));
   y -= lineH;

   if( track->charge() > 0 )
      latex->DrawLatex( x, y, " charge = +1" );
   else
      latex->DrawLatex( x, y, " charge = -1" );
   y -= lineH;
   y -= lineH;

   m_legend->SetY2(y);
   m_legend->Draw();
   m_legend = 0; // Deleted together with TPad.
}

void
FWPFCandidateDetailView::plotEtChanged()
{
   printf("plotEt = %d \n", m_plotEt);
   m_plotEt = !m_plotEt; 
   buildGLEventScene();
}

void
FWPFCandidateDetailView::rnrHcalChanged()
{
   printf("rnrHcal = %d \n", m_rnrHcal);
   m_rnrHcal = !m_rnrHcal; 
   buildGLEventScene();
}

//______________________________________________________________________________

void
FWPFCandidateDetailView::build (const FWModelId &id, const reco::PFCandidate* candidate)
{       
   m_candidate = candidate;

   // ROOT GUI 
   //
   {
      TGCompositeFrame* f = new TGVerticalFrame(m_guiFrame);
      m_guiFrame->AddFrame(f);
      f->AddFrame(new TGLabel(f, "Rng:"), new TGLayoutHints(kLHintsLeft, 2, 2, 0, 0));
      m_slider = new TGHSlider(f, 120, kSlider1 | kScaleNo);
      f->AddFrame(m_slider, new TGLayoutHints(kLHintsTop | kLHintsLeft, 2, 2, 1, 4));
      m_slider->SetRange(1, 50);
      m_slider->SetPosition(8);

      m_sliderListener = new FWIntValueListener();
      TQObject::Connect(m_slider, "PositionChanged(Int_t)", "FWIntValueListenerBase", m_sliderListener, "setValue(Int_t)");
      m_sliderListener->valueChanged_.connect(boost::bind(&FWPFCandidateDetailView::rangeChanged,this,_1));
      {
         CSGAction* action = new CSGAction(this, "Scale Et");
         TGCheckButton* b = new TGCheckButton(m_guiFrame, action->getName().c_str() );
         b->SetState(kButtonDown, true);
         m_guiFrame->AddFrame(b, new TGLayoutHints( kLHintsNormal, 2, 3, 1, 4));
         TQObject::Connect(b, "Clicked()", "CSGAction", action, "activate()");
         action->activated.connect(sigc::mem_fun(this, &FWPFCandidateDetailView::plotEtChanged));
      } 
      {
         CSGAction* action = new CSGAction(this, "RnrHcal");
         TGCheckButton* b = new TGCheckButton(m_guiFrame, action->getName().c_str() );
         b->SetState(kButtonDown, true);
         m_guiFrame->AddFrame(b, new TGLayoutHints( kLHintsNormal, 2, 3, 1, 4));
         TQObject::Connect(b, "Clicked()", "CSGAction", action, "activate()");
         action->activated.connect(sigc::mem_fun(this, &FWPFCandidateDetailView::rnrHcalChanged));
      }

   }
   makeLegend();
   setTextInfo(id, candidate);

   TGCompositeFrame* p = (TGCompositeFrame*)m_guiFrame->GetParent();
   p->MapSubwindows();
   p->Layout();

   ///////////////
   // GL stuff 


   m_candidate = candidate;

   try {
      const edm::EventBase* event = FWGUIManager::getGUIManager()->getCurrentEvent();
      edm::Handle<std::vector<reco::PFRecHit> > ecalH; 
      event->getByLabel(edm::InputTag("particleFlowRecHitECAL"), ecalH);
      if (ecalH.product()) voteMaxEtEVal(ecalH.product());

      edm::Handle<std::vector<reco::PFRecHit> > hcalH; 
      event->getByLabel(edm::InputTag("particleFlowRecHitHCAL"),hcalH);
      if (hcalH.product()) voteMaxEtEVal(hcalH.product());

      edm::Handle<std::vector<reco::PFRecHit> > hoH; 
      event->getByLabel(edm::InputTag("particleFlowRecHitHO"),hoH);
      if (hoH.product()) voteMaxEtEVal(hoH.product());

      edm::Handle<std::vector<reco::PFRecHit> > hfH; 
      event->getByLabel(edm::InputTag("particleFlowRecHitHF"),hfH);
      if (hfH.product()) voteMaxEtEVal(hfH.product());


   }
   catch(const cms::Exception& iE) {
      std::cerr << iE.what();
   }

   m_eveScene->GetGLScene()->SetSelectable(false);
   m_eventList = new TEveElementList("PFDetailView");
   m_eveScene->AddElement(m_eventList);


   viewerGL()->SetStyle(TGLRnrCtx::kOutline);
   viewerGL()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);

   TGLCameraOverlay* co = viewerGL()->GetCameraOverlay();
   co->SetShowOrthographic(kTRUE);
   co->SetOrthographicMode(TGLCameraOverlay::kAxis);

   viewerGL()->ResetCamerasAfterNextUpdate();
   try {
      buildGLEventScene();
   } 
   catch (...) {
      printf("unknown exception \n");
   }

   viewerGL()->UpdateScene(kFALSE);

   gEve->Redraw3D();

   // gEve->AddToListTree(m_eventList, true);//debug, used with --eve option
}


//______________________________________________________________________________

void FWPFCandidateDetailView::voteMaxEtEVal( const std::vector<reco::PFRecHit> *hits)
{
   if (!hits) return;

   for (std::vector<reco::PFRecHit>::const_iterator it = hits->begin(); it != hits->end(); ++it)
   {
      TEveVector centre(it->position().x(), it->position().y(), it->position().z());
      float E = it->energy();
      float Et  = FWPFMaths::calculateEt( centre, E );
      item()->context().voteMaxEtAndEnergy(Et , E );
   }
}


//______________________________________________________________________________

void FWPFCandidateDetailView::addTracks( const std::vector<reco::PFRecTrack> *tracks)
{
   for (std::vector<reco::PFRecTrack>::const_iterator it = tracks->begin(); it != tracks->end(); ++it)
   {
      /// AMT trackRef() is a collection !!!
      /*
      if (!isPntInRng(it->trackRef().innerMomentum().Eta(), it->position().Phi()))
         continue;

      TEveLine* line = new TEveLine("Track");
      line->SetMainColor(kYellow);
      int  N = it->nTrajectoryPoints();

      for (int p = 0 ; p<N; ++p) {
         pos = track.extrapolatedPoint(p).position();

         if( pos.Eta() !=0 and pos.Phi() !=0)
            line->SetNextPoint(pos.Eta(), pos.Phi(), 0);
      }
      m_eventList->AddElement(line);
      */

   }
}

//______________________________________________________________________________

void FWPFCandidateDetailView::addClusters( const std::vector<reco::PFCluster> *cluster)
{
   if (!cluster) return;

   Color_t col = kViolet+9;

   TEveStraightLineSet* ls = new TEveStraightLineSet("cluster_ls");
   ls->SetMainColor(col);
   m_eventList->AddElement(ls);
  
   TEvePointSet* ps = new TEvePointSet("cluster_ps");
   ps->SetMainColor(col);
   ps->SetMarkerStyle(2);
   ps->SetMarkerSize(0.005);
   m_eventList->AddElement(ps);
  
   for (std::vector<reco::PFCluster>::const_iterator it = cluster->begin(); it != cluster->end(); ++it)
   {
      if (!isPntInRng(it->position().Eta(), it->position().Phi()))
         continue;

      ps->SetNextPoint(it->position().Eta(), it->position().Phi(), 0);

      /*
      const std::vector< reco::PFRecHitFraction >& fractions = it->recHitFractions();
      for (std::vector< reco::PFRecHitFraction >::const_iterator fi = fractions.begin(); fi != fractions.end(); ++fi)
      {
         // !!! AMT can't get  fi->recHitRef().position()
         //    ls->AddLine(it->position().Eta(), it->position().Phi(), 0,
         //      fi->recHitRef().position().Eta(), fi->recHitRef().position().Phi(), 0);
      }
      */

   }
}
namespace {
void WrapTwoPi(std::vector<TEveVector>& hc, float y)
{

   if (TMath::Abs(hc[0].fY) < 2)
      return;


   if (hc[0].fY > 0 && hc[1].fY > 0 &&  hc[2].fY > 0 && hc[3].fY > 0 )
      return;
   if (hc[0].fY < 0 && hc[1].fY < 0 &&  hc[2].fY < 0 && hc[3].fY < 0 )
      return;


   for (int i = 0; i < 4; ++i)
      if (y > 0 && hc[i].fY < 0) 
         hc[i].fY += TMath::TwoPi();
      else if (y < 0 && hc[i].fY > 0)
         hc[i].fY -= TMath::TwoPi();

}
}
//______________________________________________________________________________
namespace 
{
TEveStraightLineSet::Line_t* AddLineToLineSet(TEveStraightLineSet* ls, const std::vector< TEveVector >& pnts, int i0, int i1) 
{
   if (0) {
      printf("add line \n");
      pnts[i0].Dump();
      pnts[i1].Dump();
   }
   return ls->AddLine(pnts[i0], pnts[i1]);  
   // return ls->AddLine(pnts[i0].Eta(),pnts[i0].Phi(), 0 , pnts[i1].Eta(),pnts[i1].Phi(), 0);
}
}
void FWPFCandidateDetailView::addHits( const std::vector<reco::PFRecHit> *hits)
{

   TEveStraightLineSet* lsOutline = ( TEveStraightLineSet*)m_eventList->FindChild("outlines");

   TEvePointSet* ps = new TEvePointSet("test");
   m_eventList->AddElement(ps);
   ps->SetMainColor(kOrange);

   for (std::vector<reco::PFRecHit>::const_iterator it = hits->begin(); it != hits->end(); ++it)
   {
      const std::vector< math::XYZPoint >& corners = it->getCornersXYZ();
      if (!isPntInRng(corners[0].eta(), corners[0].phi()))
         continue;
     
      std::vector<TEveVector> hc;
      for (int k = 0; k < 4; ++k) {
         hc.push_back(TEveVector(corners[k].eta(), corners[k].phi(), 0));
         // ps->SetNextPoint(corners[k].eta(),corners[k].phi(),0 ); //debug
      }

      WrapTwoPi(hc,  corners[0].phi());

      AddLineToLineSet(lsOutline, hc, 0, 1);
      AddLineToLineSet(lsOutline, hc, 1, 2);
      AddLineToLineSet(lsOutline, hc, 2, 3);
      AddLineToLineSet(lsOutline, hc, 3, 0);


      // get scaled corners
      TEveVector centerOfGravity = hc[0] +  hc[1] + hc[2] +  hc[3];
      centerOfGravity *= 0.25;

      std::vector<TEveVector> radialVectors;
      for (int k = 0; k < 4; ++k) 
         radialVectors.push_back(TEveVector(hc[k] - centerOfGravity));

      float factor = 1;
      if (m_plotEt) {
         float Et  = FWPFMaths::calculateEt( TEveVector(corners[0].x(), corners[0].y(), corners[0].z()), it->energy());
         factor = Et/context().getMaxEnergyInEvent(m_plotEt); 
      }
      else
         factor = it->energy()/context().getMaxEnergyInEvent(false);


      std::vector<TEveVector> scaledCorners;
      for (int k = 0; k < 4; ++k) {
         radialVectors[k] *= factor;
         scaledCorners.push_back(TEveVector(radialVectors[k] + centerOfGravity));
      }

      TEveStraightLineSet* ls = ( TEveStraightLineSet*)m_eventList->FindChild(Form("%d_rechit", it->depth() ));
      AddLineToLineSet(ls, scaledCorners, 0, 1);
      AddLineToLineSet(ls, scaledCorners, 1, 2);
      AddLineToLineSet(ls, scaledCorners, 2, 3);
      // AddLineToLineSet(ls, scaledCorners, 3, 0);
      TEveStraightLineSet::Line_t*  li =  AddLineToLineSet(ls, scaledCorners, 3, 0);
      ls->AddMarker(centerOfGravity, li->fId);    

   }
}


//______________________________________________________________________________


void FWPFCandidateDetailView::buildGLEventScene()
{
   if (m_eventList->HasChildren()) m_eventList->DestroyElements();


   for (int depth = 0; depth < 6; ++depth)
   {
      TEveStraightLineSet* ls = new TEveStraightLineSet(Form("%d_rechit", depth));

      if      (depth == 0 ) ls->SetLineColor(kGray);
      else if (depth == 1 ) ls->SetLineColor(kRed);
      else if (depth == 2 ) ls->SetLineColor(kGreen);
      else if (depth == 3 ) ls->SetLineColor(kMagenta);
      else if (depth == 4 ) ls->SetLineColor(kOrange);
      else if (depth == 5 ) ls->SetLineColor(kYellow);

      ls->SetMarkerStyle(1);
      m_eventList->AddElement(ls);
   }

   TEveStraightLineSet* ls = new TEveStraightLineSet("outlines");
   ls->SetLineColor(kGray);
   ls->SetMainTransparency(80);
   m_eventList->AddElement(ls);


   const edm::EventBase* event = FWGUIManager::getGUIManager()->getCurrentEvent();


   //
   // recHits
   //
   try {
      edm::Handle<std::vector<reco::PFRecHit> > ecalH; 
      event->getByLabel(edm::InputTag("particleFlowRecHitECAL"), ecalH);
      addHits(ecalH.product());
   }
   catch(const cms::Exception& iE) {
      std::cerr << iE.what();
   }

   if (m_rnrHcal) {
      try {
         edm::Handle<std::vector<reco::PFRecHit> > hfH; 
         event->getByLabel(edm::InputTag("particleFlowRecHitHF"), hfH);
         addHits(hfH.product());
      }
      catch(const cms::Exception& iE) {
         std::cerr << iE.what();
      }


      try {
         edm::Handle<std::vector<reco::PFRecHit> > hcalH; 
         event->getByLabel(edm::InputTag("particleFlowRecHitHBHE"),hcalH);
         addHits(hcalH.product());
      }
      catch(const cms::Exception& iE) {
         std::cerr << iE.what();
      }

      try {
         edm::Handle<std::vector<reco::PFRecHit> > hcalH; 
         event->getByLabel(edm::InputTag("particleFlowRecHitHO"),hcalH);
         addHits(hcalH.product());
      }
      catch(const cms::Exception& iE) {
         std::cerr << iE.what();
      
      }
   }


   //
   // clusters
   //
   try {
      edm::Handle<std::vector<reco::PFCluster> > ecalClustersH;
      event->getByLabel(edm::InputTag("particleFlowClusterECAL"), ecalClustersH);
      addClusters(ecalClustersH.product());
   }
   catch (const cms::Exception& iE) {
      std::cerr << iE.what();
   }

   if (m_rnrHcal) {

   try {
      edm::Handle<std::vector<reco::PFCluster> > hcalClustersH;
      event->getByLabel(edm::InputTag("particleFlowClusterHCAL"), hcalClustersH);
      addClusters(hcalClustersH.product());
   }
   catch (const cms::Exception& iE) {
      std::cerr << iE.what();
   }

   try {
      edm::Handle<std::vector<reco::PFCluster> > hcalClustersH;
      event->getByLabel(edm::InputTag("particleFlowClusterHO"), hcalClustersH);
      addClusters(hcalClustersH.product());
   }
   catch (const cms::Exception& iE) {
      std::cerr << iE.what();
   }



   }

   //
   // tracks
   //
   try {
      edm::Handle<std::vector<reco::PFRecTrack> > trackH; 
      event->getByLabel(edm::InputTag("pfTrack"),trackH);
      addTracks(trackH.product());
   }
   catch (const cms::Exception& iE) {
      std::cerr << iE.what();
   }

}

REGISTER_FWDETAILVIEW(FWPFCandidateDetailView, reco::PFCandidate, particleFlowRecHitECAL&particleFlowRecHitHF&particleFlowClusterECAL);
