/** \class CosmicMuonValidator
 *
 *  compare reconstructed and simulated cosmic muons
 *
 *  the validator assumes single muon events
 *
 *  $Date: 2011/12/22 20:57:21 $
 *  $Revision: 1.11 $
 *  \author Chang Liu   -  Purdue University <Chang.Liu@cern.ch>
 */

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

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
#include <TSystem.h>
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

class CosmicMuonValidator : public edm::EDAnalyzer {
   public:
      explicit CosmicMuonValidator(const edm::ParameterSet&);
      ~CosmicMuonValidator();

   private:

      virtual void beginJob();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      virtual void endJob();

      reco::Track bestTrack(const reco::TrackCollection&) const;

      PSimHitContainer matchedHit(const GlobalPoint&, const PSimHitContainer&) const;

      TrajectoryStateOnSurface updatedState(const TrajectoryStateOnSurface&, const PSimHit&) const;

      edm::ESHandle<Propagator> propagator() const;

      edm::InputTag trackLabel_;
      edm::InputTag simTrackLabel_;

      MuonServiceProxy* theService;

      int theDrawOption;

      int nEvent;
      int successR;
      int nNoSignal;

      TH2F* h_innerPosXY;
      TH2F* h_innerPosEP;

      TH2F* h_outerPosXY;
      TH2F* h_outerPosEP;

      TH1F* h_res;
      TH1F* h_theta;
      TH1F* h_phi;
      TH1F* h_pt_rec_sim; 
      TH1F* h_phi_rec_sim;
      TH1F* h_theta_rec_sim;
      TH1F* h_Pres_inv_sim;
      TH1F* h_pt_sim;
      TH1F* h_pt_rec;
      TH1F* hnhit;

      TH1F* htotal4D;
      TH1F* htotalSeg;


};

CosmicMuonValidator::CosmicMuonValidator(const edm::ParameterSet& iConfig)
{

  trackLabel_ = iConfig.getParameter<edm::InputTag>("TrackLabel");
  simTrackLabel_ = iConfig.getParameter<edm::InputTag>("SimTrackLabel");
  theDrawOption = iConfig.getUntrackedParameter<int>("DrawOption", 1); 

  // service parameters
  edm::ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");
  // the services
  theService = new MuonServiceProxy(serviceParameters);

  nEvent = 0;
  successR = 0;
  nNoSignal = 0;
  
}


CosmicMuonValidator::~CosmicMuonValidator()
{
  if (theService) delete theService;

}


void CosmicMuonValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   theService->update(iSetup);

   nEvent++;
   std::cout << "reading event " << nEvent << std::endl;

   Handle<reco::TrackCollection> muons;
   iEvent.getByLabel(trackLabel_,muons);
   cout << "cosmic Muon: " <<muons->size() <<endl;

 //  if (muons->empty()) return;
   if ( !muons->empty())  successR++;
   
   float ptsim = 0; 
   float simC = 0; 
   float thetasim = 0;
   float phisim = 0;

   Handle<edm::SimTrackContainer> simTracks;
   iEvent.getByLabel(simTrackLabel_,simTracks);
   cout << "simTracks: " <<simTracks->size() <<endl;

   for (SimTrackContainer::const_iterator simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){
    if (abs((*simTrack).type()) == 13) {
       cout << "MC Muon: mom (" << simTrack->momentum().x() << "," << simTrack->momentum().y() << "," <<simTrack->momentum().z()
 << ")"<<endl;
       thetasim = simTrack->momentum().theta();
       phisim = simTrack->momentum().phi();

       simC = - simTrack->type()/13.;
       ptsim = simTrack->momentum().pt();
    }
  }

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
   reco::Track muon = bestTrack(*muons);

   cout << "cosmic Muon Track: " 
        << " mom: (" << muon.px() << "," << muon.py() << "," << muon.pz()
        << ")"<<endl;

   if ( fabs(muon.p()) < 1e-5 ) return; //prevent those failed to extrapolation to vertex

   math::XYZVector innerMo = muon.innerMomentum();

   float ptreco = muon.pt();
   int qreco = muon.charge();
 
   h_pt_rec->Fill(ptreco);
   hnhit->Fill(muon.recHitsSize());

   cout<<"resolution "<<(qreco/ptreco-simC/ptsim)*ptsim/simC<<endl;

   h_res->Fill((qreco/ptreco-simC/ptsim)*ptsim/simC);

   GlobalVector im(innerMo.x(),innerMo.y(),innerMo.z());
   float thetareco = im.theta();
   float phireco = im.phi();
   h_theta->Fill(Geom::Theta<float>(thetareco-thetasim));
   h_phi->Fill(Geom::Phi<float>(phireco-phisim));

   math::XYZPoint innerPo = muon.innerPosition();
   GlobalPoint ip(innerPo.x(), innerPo.y(),innerPo.z());

   h_innerPosXY->Fill(ip.x(), ip.y());
   h_innerPosEP->Fill(ip.eta(), Geom::Phi<float>(ip.phi()));

   math::XYZPoint outerPo = muon.outerPosition();
   GlobalPoint op(outerPo.x(), outerPo.y(),outerPo.z());

   h_outerPosXY->Fill(op.x(), op.y());
   h_outerPosEP->Fill(op.eta(), Geom::Phi<float>(op.phi()));

  //Now compare innermost state with associated sim hit

  Handle<PSimHitContainer>  dtSimHits;
  Handle<PSimHitContainer>  cscSimHits;
  Handle<PSimHitContainer>  rpcSimHits;

  iEvent.getByLabel("g4SimHits","MuonDTHits", dtSimHits);
  iEvent.getByLabel("g4SimHits","MuonCSCHits", cscSimHits);
  iEvent.getByLabel("g4SimHits","MuonRPCHits", rpcSimHits);

  cout<<"DT simHits collections: "<<dtSimHits->size()<<endl;
  cout<<"CSC simHits collections: "<<cscSimHits->size()<<endl;
  cout<<"RPC simHits collections: "<<rpcSimHits->size()<<endl;

  PSimHitContainer allSimHits = *dtSimHits;
  allSimHits.insert(allSimHits.end(),(cscSimHits)->begin(), (cscSimHits)->end());
  allSimHits.insert(allSimHits.end(),(rpcSimHits)->begin(), (rpcSimHits)->end());

   cout<<"allSimHits "<<allSimHits.size()<<endl;

   if ( allSimHits.empty() ) return;

   PSimHitContainer  msimh = matchedHit(ip, allSimHits);

   if ( !msimh.empty() ) {

     DetId idSim( msimh.front().detUnitId() );

     GlobalVector simmom = theService->trackingGeometry()->idToDet(idSim)->surface().toGlobal(msimh.front().momentumAtEntry());

     

     TrajectoryStateOnSurface innerTSOS = trajectoryStateTransform::innerStateOnSurface(muon,*theService->trackingGeometry(),&*theService->magneticField());
  
     TrajectoryStateOnSurface stateAH = updatedState(innerTSOS,msimh.front());
     if (!stateAH.isValid()) return;
     im = stateAH.globalMomentum();

     cout<<"sim Momentum: "<<simmom<<endl;
     cout<<"track Mom here: "<<im<<endl;

     h_pt_rec_sim->Fill( ((double)simmom.perp()) - im.perp());
     h_phi_rec_sim->Fill( ( (Geom::Phi<float>(simmom.phi())) - Geom::Phi<float>(im.phi())) * 180/acos(-1.));

     h_Pres_inv_sim->Fill( (1/im.perp() - 1/((double)simmom.perp())) / (1/((double)simmom.perp())));

     h_pt_sim->Fill(((double)simmom.perp()));

     h_theta_rec_sim->Fill( ( ((double)simmom.theta())-im.theta()) * 180/acos(-1.));

    }
}

void CosmicMuonValidator::beginJob()
{

  cout<<"Prepare histograms "<<"\n";
  edm::Service<TFileService> fs;

  h_res = fs->make<TH1F>("h_res","resolution of P_{T}",50,-5.0,5.0);

  h_theta = fs->make<TH1F>("h_theta","theta angle ",50,-0.1,0.1);
  h_phi = fs->make<TH1F>("h_phi","phi angle ",50,-2.0,2.0);

  hnhit = fs->make<TH1F>("hnhit","Number of Hits in Track by Cos",60,0.0,60.0);

  h_innerPosXY = fs->make<TH2F>("h_innerPosXY", "inner x-y", 100, -700.0, 700.0, 100, -700.0, 700.0);
  h_innerPosEP = fs->make<TH2F>("h_innerPosEP", "inner #eta-#phi", 100, -2.4, 2.4, 100, -3.3, 3.3);

  h_outerPosXY = fs->make<TH2F>("h_outerPosXY", "outer x-y", 100, -700.0, 700.0, 100, -700.0, 700.0);
  h_outerPosEP = fs->make<TH2F>("h_outerPosEP", "outer #eta-#phi", 100, -2.4, 2.4, 100, -3.3, 3.3);

  h_pt_rec_sim = fs->make<TH1F>("h_pt_res_sim","diff of P_{T} at SimHit",50,-2.0,2.0);
  h_phi_rec_sim = fs->make<TH1F>("h_phi_res_sim","diff of #phi at SimHit",50,-2.0,2.0);
  h_theta_rec_sim = fs->make<TH1F>("h_theta_res_sim","diff of #theta at SimHit",50,-2.0,2.0);
  h_Pres_inv_sim = fs->make<TH1F>("h_Pres_inv_sim","resolution of P_{T} at SimHit",70,-1.0,1.0);

  h_pt_sim = fs->make<TH1F>("h_pt_sim","distribution of P_{T} at SimHit",100,0.0,100.0);
  h_pt_rec = fs->make<TH1F>("h_pt_rec","distribution of P_{T} at SimHit",100,0.0,100.0);
  htotal4D = fs->make<TH1F>("htotal4D","# of Segments",15,0.0,15.0);
  htotalSeg = fs->make<TH1F>("htotalSeg","# of Segments",15,0.0,15.0);

}

void CosmicMuonValidator::endJob() {

  float eff = (float)successR/((float)nEvent-(float)nNoSignal) * 100 ;

  std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << "Analyzed " << nEvent << " events, " << std::endl;
  std::cout << successR<< " events are successfully reconstructed. "<< std::endl;
  std::cout << nNoSignal<< " events do not have good enough signals. "<< std::endl;
  std::cout << "Reconstruction efficiency is approximately "<< eff << "%. "<< std::endl;
  std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;

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

  h_res->SetXTitle("(q^{reco}/P^{reco}_{T}-q^{simTrack}/P^{simTrack}_{T})/(q^{simTrack}/P^{simTrack}_{T})");
  h_res->SetLineWidth(2);
  h_res->SetLineColor(2);
  h_res->SetLineStyle(1);

  h_theta->SetXTitle("#theta^{reco}-#theta^{simTrack}");
  h_theta->SetLineWidth(2);
  h_theta->SetLineColor(2);
  h_theta->SetLineStyle(1);

  h_phi->SetXTitle("#phi^{reco}-#phi^{simTrack}");
  h_phi->SetLineWidth(2);
  h_phi->SetLineColor(2);
  h_phi->SetLineStyle(1);


  h_pt_rec_sim->SetXTitle("P^{simHit}_{T}-P^{reco}_{T}");
  h_pt_rec_sim->SetLineWidth(2);
  h_pt_rec_sim->SetLineColor(2);
  h_pt_rec_sim->SetLineStyle(1);

  h_phi_rec_sim->SetXTitle("#phi^{simHit}-#phi^{reco}");
  h_phi_rec_sim->SetLineWidth(2);
  h_phi_rec_sim->SetLineColor(2);
  h_phi_rec_sim->SetLineStyle(1);

  h_theta_rec_sim->SetXTitle("#theta^{simHit}-#theta^{reco}");
  h_theta_rec_sim->SetLineWidth(2);
  h_theta_rec_sim->SetLineColor(2);
  h_theta_rec_sim->SetLineStyle(1);

  h_Pres_inv_sim->SetXTitle("(q^{reco}/P^{reco}_{T}-q^{simHit}/P^{simHit}_{T})/(q^{simHit}/P^{simHit}_{T})");
  h_Pres_inv_sim->SetLineWidth(2);
  h_Pres_inv_sim->SetLineColor(2);
  h_Pres_inv_sim->SetLineStyle(1);

  hnhit->SetXTitle("N_{RecHits}");
  hnhit->SetLineWidth(2);
  hnhit->SetLineColor(2);
  hnhit->SetLineStyle(1);

  h_pt_sim->SetXTitle("P_{t}");
  h_pt_sim->SetLineWidth(2);
  h_pt_sim->SetLineColor(4);
  h_pt_rec->SetLineWidth(2);
  h_pt_rec->SetLineColor(2);
  h_pt_sim->SetLineStyle(1);
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


    TCanvas* cRes1 = new TCanvas("TrackPtRes","Resolution of P_{T} wrt SimTrack",10,10,800,600);
    cRes1->SetFillColor(0);
    cRes1->SetGrid(1);
    cRes1->SetRightMargin(0.03);
    cRes1->SetTopMargin(0.02);
    cRes1->cd();
    h_res->DrawCopy("HE");
    cRes1->Update();
    cRes1->Write();

    TCanvas* cRes2 = new TCanvas("TrackTheta","Resolution of Theta wrt SimTrack",10,10,800,600);
    cRes2->SetFillColor(0);
    cRes2->SetGrid(1);
    cRes2->SetRightMargin(0.03);
    cRes2->SetTopMargin(0.02);
    cRes2->cd();
    h_theta->DrawCopy("HE");
    cRes2->Update();
    cRes2->Write();

    TCanvas* cRes3 = new TCanvas("TrackPhi","Resolution of phi wrt SimTrack",10,10,800,600);
    cRes3->SetFillColor(0);
    cRes3->SetGrid(1);
    cRes3->SetRightMargin(0.03);
    cRes3->SetTopMargin(0.02);
    cRes3->cd();
    h_phi->DrawCopy("HE");
    cRes3->Update();
    cRes3->Write();

    TCanvas* cRes4 = new TCanvas("inTsosPtDiff","Resolution of P_{T} at SimHit",10,10,800,600);
    cRes4->SetFillColor(0);
    cRes4->SetGrid(1);
    cRes4->SetRightMargin(0.03);
    cRes4->SetTopMargin(0.02);
    cRes4->cd();
    h_pt_rec_sim->DrawCopy("HE");
    cRes4->Update();
    cRes4->Write();

    TCanvas* cRes5 = new TCanvas("inTsosPhi","Resolution of Phi at SimHit",10,10,800,600);
    cRes5->SetFillColor(0);
    cRes5->SetGrid(1);
    cRes5->SetRightMargin(0.03);
    cRes5->SetTopMargin(0.02);
    cRes5->cd();
    h_phi_rec_sim->DrawCopy("HE");
    cRes5->Update();
    cRes5->Write();

    TCanvas* cRes6 = new TCanvas("inTsosTheta","Resolution of #theta at SimHit",10,10,800,600);
    cRes6->SetFillColor(0);
    cRes6->SetGrid(1);
    cRes6->SetRightMargin(0.03);
    cRes6->SetTopMargin(0.02);
    cRes6->cd();
    h_theta_rec_sim->DrawCopy("HE");
    cRes6->Update();
    cRes6->Write();

    TCanvas* cRes7 = new TCanvas("inTsosPtRes","Resolution of P_{T} at SimHit",10,10,800,600);
    cRes7->SetFillColor(0);
    cRes7->SetGrid(1);
    cRes7->SetRightMargin(0.03);
    cRes7->SetTopMargin(0.02);
    cRes7->cd();
    h_Pres_inv_sim->DrawCopy("HE");
    TF1* g2 = new TF1("g2","gaus",-0.5,0.5);
    g2->SetLineColor(4);
    h_Pres_inv_sim->Fit("g2","R");

    cRes7->Update();
    cRes7->Write();

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

    h_pt_sim->DrawCopy("HE");
    h_pt_rec->DrawCopy("HEsame");
    TLegend* legend8 = new TLegend(0.7,0.6,0.9,0.8);
    legend8->SetTextAlign(32);
    legend8->SetTextColor(1);
    legend8->SetTextSize(0.04);
    legend8->AddEntry("h_pt_sim","By Sim","l");
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

    ttl.DrawLatex(.4, .95, "resolution wrt simTrack (pt, theta, phi) ");
    page_print.str(""); page_print << page + 1;
    ttl.DrawLatex(.9, .02, page_print.str().c_str());

    pad[page]->Draw();
    pad[page]->Divide(1, 3);
    pad[page]->cd(1);
    h_res->Draw("HE");

    pad[page]->cd(2);
    h_res->Draw("HE");

    pad[page]->cd(3);
    h_phi->Draw("HE");


    page++;
    cpdf->Update();

    pdf->NewPage();
    cpdf->Clear();
    cpdf->cd(0);

    ttl.DrawLatex(.4, .95, "resolution wrt simHit at innermost (pt, theta, phi) ");
    page_print.str(""); page_print << page + 1;
    ttl.DrawLatex(.9, .02, page_print.str().c_str());
    pad[page]->Draw();
    pad[page]->Divide(1, 3);
    pad[page]->cd(1);
    h_Pres_inv_sim->Draw("HE");
    pad[page]->cd(2);
    h_theta_rec_sim->Draw("HE");
    pad[page]->cd(3);
    h_phi_rec_sim->Draw("HE");

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
    pad[page]->cd(2);
    h_pt_sim->Draw("HE");

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

reco::Track 
CosmicMuonValidator::bestTrack(const reco::TrackCollection& muons) const {

   reco::Track bestOne = muons.front();

   for(reco::TrackCollection::const_iterator muon = muons.begin()+1; muon != muons.end(); ++ muon ) {

    if (( (*muon).found() > bestOne.found() ) ||
       (((*muon).found() == bestOne.found()) && ((*muon).chi2() < bestOne.chi2())) )
       bestOne = (*muon);
   }
   return bestOne;

}


PSimHitContainer CosmicMuonValidator::matchedHit(const GlobalPoint& tp, const PSimHitContainer& simHs) const {

      float dcut = 3.0;
      PSimHitContainer result;
      PSimHit rs = simHs.front();
      bool hasMatched = false;

      if (simHs.empty()) return result;

      for (PSimHitContainer::const_iterator ish = simHs.begin();
           ish != simHs.end(); ish++ ) {

            if (abs( (*ish).particleType() ) != 13 ) continue;

            DetId idsim( (*ish).detUnitId() );

            GlobalPoint sp = theService->trackingGeometry()->idToDet(idsim)->surface().toGlobal(ish->entryPoint()); //entryPoint or localPosition??
            GlobalVector dist = sp - tp;
            float d = fabs(dist.y());
            if ( d < dcut ) {
               rs = (*ish);
               dcut = d;
               hasMatched = true;
             }
       }
      if ( hasMatched ) { 
            result.push_back(rs);
            DetId idsim( rs.detUnitId() );
            cout<<"selected simhit: "<< theService->trackingGeometry()->idToDet(idsim)->surface().toGlobal(rs.entryPoint())<<endl;
            cout<<"matched with   : "<< tp <<endl;

      }
      return result;

}

TrajectoryStateOnSurface CosmicMuonValidator::updatedState(const TrajectoryStateOnSurface& tsos,const PSimHit& hit) const{

    DetId idsim( hit.detUnitId() );

    TrajectoryStateOnSurface  result = propagator()->propagate(tsos,theService->trackingGeometry()->idToDet(idsim)->surface());

    return result;
}

edm::ESHandle<Propagator> CosmicMuonValidator::propagator() const {
   return theService->propagator("SteppingHelixPropagatorAny");
}

