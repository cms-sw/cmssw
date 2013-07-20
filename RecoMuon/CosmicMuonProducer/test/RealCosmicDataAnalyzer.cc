/** \class RealCosmicDataAnalyzer
 *
 *  $Date: 2010/02/25 00:32:41 $
 *  $Revision: 1.4 $
 *  \author Chang Liu   -  Purdue University <Chang.Liu@cern.ch>
 */

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <iostream>

#include <TH1D.h>
#include <TTree.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TFrame.h>
#include <TMath.h>
#include <TF1.h>
#include <TPostScript.h>
#include <TPad.h>
#include <TText.h>
#include <TLatex.h>

using namespace std;
using namespace edm;

class RealCosmicDataAnalyzer : public edm::EDAnalyzer {
   public:
      explicit RealCosmicDataAnalyzer(const edm::ParameterSet&);
      ~RealCosmicDataAnalyzer();

   private:

      virtual void beginJob();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      virtual void endJob();

      edm::ESHandle<Propagator> propagator() const;

      edm::InputTag trackLabel_;

      MuonServiceProxy* theService;

      int theDrawOption;

      int nEvent;
      int successR;
      int nNoSignal;

      TH2F* h_innerPosXY;
      TH2F* h_innerPosEP;

      TH2F* h_outerPosXY;
      TH2F* h_outerPosEP;

      TH1F* h_inOutDis;

      TH1F* h_theta;
      TH1F* h_phi;
      TH1F* h_pt_rec;
      TH1F* hnhit;

      TH1F* htotal4D;
      TH1F* htotalSeg;


};

RealCosmicDataAnalyzer::RealCosmicDataAnalyzer(const edm::ParameterSet& iConfig)
{

  trackLabel_ = iConfig.getParameter<edm::InputTag>("TrackLabel");
  theDrawOption = iConfig.getUntrackedParameter<int>("DrawOption", 1); 

  // service parameters
  edm::ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");
  // the services
  theService = new MuonServiceProxy(serviceParameters);

  nEvent = 0;
  successR = 0;
  nNoSignal = 0;
  
}


RealCosmicDataAnalyzer::~RealCosmicDataAnalyzer()
{
  if (theService) delete theService;

}


void RealCosmicDataAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   theService->update(iSetup);

   nEvent++;
   std::cout << "reading event " << nEvent << std::endl;

   Handle<reco::TrackCollection> muons;
   iEvent.getByLabel(trackLabel_,muons);
   cout << "cosmic Muon: " <<muons->size() <<endl;

 //  if (muons->empty()) return;
   if ( !muons->empty())  successR++;
   
   // Get Segment collections from the Event
   edm::Handle<DTRecSegment4DCollection> dtSegments;
   iEvent.getByLabel("dt4DSegments", dtSegments);

   edm::Handle<CSCSegmentCollection> cscSegments;
   iEvent.getByLabel("cscSegments", cscSegments);

   int nSeg = dtSegments->size() + cscSegments->size();
   cout<<"cscSegments: "<<cscSegments->size()<<endl;

   htotalSeg->Fill(nSeg);

   int n4D = 0;

   for (DTRecSegment4DCollection::const_iterator idt = dtSegments->begin();
        idt != dtSegments->end(); ++idt) {
        if (idt->dimension() < 4) continue;
        bool sameChamber = false;
        DetId thisId = idt->geographicalId();

        for (DTRecSegment4DCollection::const_iterator idt2 = dtSegments->begin();
             idt2 != idt; ++idt2) {
               if (idt2->geographicalId() == thisId )   sameChamber = true;   
        }
        if (!sameChamber) n4D++;
   }

   for (CSCSegmentCollection::const_iterator icsc = cscSegments->begin();
        icsc != cscSegments->end(); ++icsc) {
        if (icsc->dimension() < 4) continue;
        bool sameChamber = false;
        DetId thisId = icsc->geographicalId();

        for (CSCSegmentCollection::const_iterator icsc2 = cscSegments->begin();
             icsc2 != icsc; ++icsc2) {
               if (icsc2->geographicalId() == thisId )   sameChamber = true;
        }

        if (!sameChamber) n4D++;
   }  

   htotal4D->Fill(n4D);
   if ( n4D < 2 ) nNoSignal++;

   if (muons->empty())  return;

   for(reco::TrackCollection::const_iterator muon = muons->begin(); muon != muons->end(); ++ muon ) {
     cout << "cosmic Muon Track: " 
          << " mom: (" << muon->px() << "," << muon->py() << "," << muon->pz()
          << ")"<<endl;
     if ( fabs(muon->p()) < 1e-5 ) return; //prevent those failed to extrapolation to vertex

     math::XYZVector innerMo = muon->innerMomentum();

     float ptreco = muon->pt();

     h_pt_rec->Fill(ptreco);

     hnhit->Fill(muon->recHitsSize());

     GlobalVector im(innerMo.x(),innerMo.y(),innerMo.z());

     h_theta->Fill(im.theta());
     h_phi->Fill(im.phi());

     math::XYZPoint innerPo = muon->innerPosition();
     GlobalPoint ip(innerPo.x(), innerPo.y(),innerPo.z());

     h_innerPosXY->Fill(ip.x(), ip.y());
     h_innerPosEP->Fill(ip.eta(), Geom::Phi<float>(ip.phi()));

     math::XYZPoint outerPo = muon->outerPosition();
     GlobalPoint op(outerPo.x(), outerPo.y(),outerPo.z());

     h_outerPosXY->Fill(op.x(), op.y());
     h_outerPosEP->Fill(op.eta(), Geom::Phi<float>(op.phi()));
     h_inOutDis->Fill((ip-op).perp());

  }

}

void RealCosmicDataAnalyzer::beginJob()
{
  edm::Service<TFileService> fs;

  h_theta = fs->make<TH1F>("h_theta","theta angle ",50,-0.1,0.1);
  h_phi = fs->make<TH1F>("h_phi","phi angle ",50,-2.0,2.0);

  hnhit = fs->make<TH1F>("hnhit","Number of Hits in Track by Cos",60,0.0,60.0);

  h_innerPosXY = fs->make<TH2F>("h_innerPosXY", "inner x-y", 100, -700.0, 700.0, 100, -700.0, 700.0);
  h_innerPosEP = fs->make<TH2F>("h_innerPosEP", "inner #eta-#phi", 100, -2.4, 2.4, 100, -3.3, 3.3);

  h_outerPosXY = fs->make<TH2F>("h_outerPosXY", "outer x-y", 100, -700.0, 700.0, 100, -700.0, 700.0);
  h_outerPosEP = fs->make<TH2F>("h_outerPosEP", "outer #eta-#phi", 100, -2.4, 2.4, 100, -3.3, 3.3);

  h_pt_rec = fs->make<TH1F>("h_pt_rec","distribution of P_{T}",100,0.0,100.0);
  h_inOutDis = fs->make<TH1F>("h_inOutDis","distribution of inner outer distance",200,0.0,1500.0);
  htotal4D = fs->make<TH1F>("htotal4D","# of Segments",15,0.0,15.0);
  htotalSeg = fs->make<TH1F>("htotalSeg","# of Segments",15,0.0,15.0);
}

void RealCosmicDataAnalyzer::endJob() {

  h_innerPosXY->SetXTitle("X");
  h_innerPosXY->SetYTitle("Y");
  h_innerPosXY->SetMarkerColor(2);
  h_innerPosXY->SetMarkerStyle(24);

  h_outerPosXY->SetXTitle("X");
  h_outerPosXY->SetYTitle("Y");
  h_outerPosXY->SetMarkerColor(2);
  h_outerPosXY->SetMarkerStyle(24);

  h_innerPosEP->SetXTitle("#eta");
  h_innerPosEP->SetYTitle("#phi");
  h_innerPosEP->SetMarkerColor(2);
  h_innerPosEP->SetMarkerStyle(24);

  h_outerPosEP->SetXTitle("#eta");
  h_outerPosEP->SetYTitle("#phi");
  h_outerPosEP->SetMarkerColor(2);
  h_outerPosEP->SetMarkerStyle(24);

  hnhit->SetXTitle("N_{RecHits}");
  hnhit->SetLineWidth(2);
  hnhit->SetLineColor(2);
  hnhit->SetLineStyle(1);

  h_pt_rec->SetLineWidth(2);
  h_pt_rec->SetLineColor(2);
  h_pt_rec->SetLineStyle(2);

  htotal4D->SetXTitle("No. of Segments");
  htotal4D->SetLineWidth(2);
  htotal4D->SetLineColor(2);

  htotalSeg->SetLineWidth(2);
  htotalSeg->SetLineColor(4);

  htotal4D->SetLineStyle(1);
  htotalSeg->SetLineStyle(2);

  int theDrawOption = 1;

  if ( theDrawOption == 0 ) {
    TCanvas* c2 = new TCanvas("innerTSOSXY","XY",10,10,800,600);
    c2->SetFillColor(0);
    c2->SetGrid(1);
    c2->SetRightMargin(0.03);
    c2->SetTopMargin(0.02);
    c2->cd();
    h_innerPosXY->Draw("SCAT");
    c2->Update();
    c2->Write();

    TCanvas* c2a = new TCanvas("outerTSOSXY","Outer XY",10,10,800,600);
    c2a->SetFillColor(0);
    c2a->SetGrid(1);
    c2a->SetRightMargin(0.03);
    c2a->SetTopMargin(0.02);
    c2a->cd();
    h_outerPosXY->Draw("SCAT");
    c2a->Update();
    c2a->Write();


    TCanvas* c3 = new TCanvas("innerEtaPhi","Inner #eta #phi",10,10,800,600);
    c3->SetFillColor(0);
    c3->SetGrid(1);
    c3->SetRightMargin(0.03);
    c3->SetTopMargin(0.02);
    c3->cd();
    h_innerPosEP->Draw("SCAT");
    c3->Update();
    c3->Write();

    TCanvas* c3a = new TCanvas("outerEtaPhi","Outer #eta #phi",10,10,800,600);
    c3a->SetFillColor(0);
    c3a->SetGrid(1);
    c3a->SetRightMargin(0.03);
    c3a->SetTopMargin(0.02);
    c3a->cd();
    h_outerPosEP->Draw("SCAT");
    c3a->Update();
    c3a->Write();

    TCanvas* cRes2 = new TCanvas("TrackTheta","Theta",10,10,800,600);
    cRes2->SetFillColor(0);
    cRes2->SetGrid(1);
    cRes2->SetRightMargin(0.03);
    cRes2->SetTopMargin(0.02);
    cRes2->cd();
    h_theta->DrawCopy("HE");
    cRes2->Update();
    cRes2->Write();

    TCanvas* cRes3 = new TCanvas("TrackPhi","phi",10,10,800,600);
    cRes3->SetFillColor(0);
    cRes3->SetGrid(1);
    cRes3->SetRightMargin(0.03);
    cRes3->SetTopMargin(0.02);
    cRes3->cd();
    h_phi->DrawCopy("HE");
    cRes3->Update();
    cRes3->Write();

    TCanvas* c7 = new TCanvas("nHits","Number of RecHits in Track",10,10,800,600);
    c7->SetFillColor(0);
    c7->SetGrid(1);
    c7->SetRightMargin(0.03);
    c7->SetTopMargin(0.02);
    c7->cd();
    hnhit->DrawCopy("HE");
    c7->Update();
    c7->Write();

    TCanvas* cRes8 = new TCanvas("PtDis","Distribution of P_{T} at SimHit",10,10,800,600);
    cRes8->SetFillColor(0);
    cRes8->SetGrid(1);
    cRes8->SetRightMargin(0.03);
    cRes8->SetTopMargin(0.02);
    cRes8->cd();

    h_pt_rec->DrawCopy("HE");
    TLegend* legend8 = new TLegend(0.7,0.6,0.9,0.8);
    legend8->SetTextAlign(32);
    legend8->SetTextColor(1);
    legend8->SetTextSize(0.04);
    legend8->AddEntry("h_pt_rec","By Cos","l");
    legend8->Draw();
    cRes8->Update();
    cRes8->Write();

    TCanvas* csegs = new TCanvas("csegs","Total & 4D Segments",10,10,800,600);

    csegs->SetFillColor(0);
    csegs->SetGrid(1);
    csegs->SetRightMargin(0.03);
    csegs->SetTopMargin(0.02);
    csegs->cd();

    htotal4D->DrawCopy("HE");
    htotalSeg->DrawCopy("HEsame");

    TLegend* legendseg = new TLegend(0.6,0.2,0.9,0.4);
    legendseg->SetTextAlign(32);
    legendseg->SetTextColor(1);
    legendseg->SetTextSize(0.04);
    legendseg->AddEntry("htotal4D","4D Segments","l");
    legendseg->AddEntry("htotalSeg","total Segments","l");
    legendseg->Draw();

    csegs->Update();
    csegs->Write();

  } else {

    TCanvas *cpdf = new TCanvas("cpdf", "", 0, 1, 500, 700);
    cpdf->SetTicks();

    TPostScript* pdf = new TPostScript("cosmicValidation.ps", 111);

    const int NUM_PAGES = 7;
    TPad *pad[NUM_PAGES];
    for (int i_page=0; i_page<NUM_PAGES; i_page++)
      pad[i_page] = new TPad("","", .05, .05, .95, .93);

    ostringstream page_print;
    int page = 0;

    TLatex ttl;

    pdf->NewPage();
    cpdf->Clear();
    cpdf->cd(0);
    ttl.DrawLatex(.4, .95, "inner and outer state (x,y) ");
    page_print.str(""); page_print << page + 1;
    ttl.DrawLatex(.9, .02, page_print.str().c_str());
    pad[page]->Draw();
    pad[page]->Divide(1, 2);
    pad[page]->cd(1);
    h_innerPosXY->Draw("SCAT");
    pad[page]->cd(2); 
    h_outerPosXY->Draw("SCAT");

    page++;
    cpdf->Update();

    pdf->NewPage();
    cpdf->Clear();
    cpdf->cd(0);
    ttl.DrawLatex(.4, .95, "inner and outer state (eta, phi) ");
    page_print.str(""); page_print << page + 1;
    ttl.DrawLatex(.9, .02, page_print.str().c_str());
    pad[page]->Draw();
    pad[page]->Divide(1, 2);
    pad[page]->cd(1);
    h_innerPosEP->Draw("SCAT");
    pad[page]->cd(2);
    h_outerPosEP->Draw("SCAT");

    page++;
    cpdf->Update();

    pdf->NewPage();
    cpdf->Clear();
    cpdf->cd(0);

    pad[page]->cd(3);
    h_phi->Draw("HE");

    page++;
    cpdf->Update();

    pdf->NewPage();
    cpdf->Clear();
    cpdf->cd(0);

    page++;
    cpdf->Update();

    pdf->NewPage();
    cpdf->Clear();
    cpdf->cd(0);

    ttl.DrawLatex(.4, .95, "number of hits ");
    page_print.str(""); page_print << page + 1;
    ttl.DrawLatex(.9, .02, page_print.str().c_str());
    pad[page]->Draw();
    pad[page]->cd();
    hnhit->Draw("HE");

    page++;
    cpdf->Update();

    pdf->NewPage();
    cpdf->Clear();
    cpdf->cd(0);

    ttl.DrawLatex(.4, .95, "pt distribution ");
    page_print.str(""); page_print << page + 1;
    ttl.DrawLatex(.9, .02, page_print.str().c_str());
    pad[page]->Draw();
    pad[page]->Divide(1, 2);
    pad[page]->cd(1);
    h_pt_rec->Draw("HE");

    page++;
    cpdf->Update();

    pdf->NewPage();
    cpdf->Clear();
    cpdf->cd(0);

    ttl.DrawLatex(.4, .95, "Number of hits ");
    page_print.str(""); page_print << page + 1;
    ttl.DrawLatex(.9, .02, page_print.str().c_str());
    pad[page]->Draw();
    pad[page]->Divide(1, 2);
    pad[page]->cd(1);
    htotal4D->Draw("HE");
    pad[page]->cd(2);
    htotalSeg->Draw("HE");

    pdf->Close(); 
  }

}

edm::ESHandle<Propagator> RealCosmicDataAnalyzer::propagator() const {
   return theService->propagator("SteppingHelixPropagatorAny");
}


//define this as a plug-in

DEFINE_FWK_MODULE(RealCosmicDataAnalyzer);
