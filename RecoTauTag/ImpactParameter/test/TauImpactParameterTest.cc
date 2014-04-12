// -*- C++ -*-
//
// Package:    TauImpactParameterTest
// Class:      TauImpactParameterTest
//
/**\class TauImpactParameterTest TauImpactParameterTest.cc RecoTauTag/ImpactParameter/test/TauImpactParameterTest.cc

 Description: EDAnalyzer to show how to get the tau impact parameter
 Implementation:

*/
//
// Original Author:  Sami Lehti
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <string>

using namespace std;
using namespace edm;
using namespace reco;

#include "TCanvas.h"
#include "TH1F.h"
#include "TTree.h"
#include "TLatex.h"
#include "TGraph.h"
#include "TH2F.h"

class TauImpactParameterTest : public edm::EDAnalyzer {

  public:
        TauImpactParameterTest(const edm::ParameterSet&);
        ~TauImpactParameterTest();

        virtual void analyze(const edm::Event&, const edm::EventSetup&);
        virtual void beginJob();
        virtual void endJob();

  private:
	TH1F* h_ip2d_1prong;
        TH1F* h_ip2d_3prong;
        TH1F* h_ip2d_3prong_leadingTrack;
        TH1F* h_sip2d_leadingTrack;

        TH1F* h_ip3d_1prong;
        TH1F* h_ip3d_3prong;
        TH1F* h_ip3d_3prong_leadingTrack;
        TH1F* h_sip3d_leadingTrack;

        TTree* t_performance;

        std::string jetTagSrc;

	int nevents;
	double ip2d,ip3d,sip2d,sip3d;
};


TauImpactParameterTest::TauImpactParameterTest(const edm::ParameterSet& iConfig){
	jetTagSrc = iConfig.getParameter<std::string>("JetTagProd");

        h_ip2d_1prong = new TH1F("h_ip2d_1prong","",50,0,1);
        h_ip2d_3prong = new TH1F("h_ip2d_3prong","",50,0,1);

	h_ip2d_3prong_leadingTrack = (TH1F*)h_ip2d_1prong->Clone("h_ip2d_3prong_leadingTrack");
	h_sip2d_leadingTrack = new TH1F("h_sip2d_leadingTrack","",100,0,20);

        h_ip3d_1prong = new TH1F("h_ip3d_1prong","",50,0,1);
        h_ip3d_3prong = (TH1F*)h_ip3d_1prong->Clone("h_ip3d_3prong");
        h_ip3d_3prong_leadingTrack = (TH1F*)h_ip3d_1prong->Clone("h_ip3d_3prong_leadingTrack");
        h_sip3d_leadingTrack = new TH1F("h_sip3d_leadingTrack","",100,0,1000);

	t_performance = new TTree("performance","");
        t_performance->Branch("ip2d",&ip2d,"ip2d/D");
        t_performance->Branch("ip3d",&ip3d,"ip3d/D");
        t_performance->Branch("sip2d",&sip2d,"sip2d/D");
        t_performance->Branch("sip3d",&sip3d,"sip3d/D");

	nevents = 0;
}

TauImpactParameterTest::~TauImpactParameterTest(){}



void TauImpactParameterTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

	nevents++;

	Handle<TauImpactParameterInfoCollection> tauHandle;
	iEvent.getByLabel(jetTagSrc,tauHandle);

	const TauImpactParameterInfoCollection & tauIpInfo = *(tauHandle.product());
	LogInfo("TauImpactParameterTest") << "Found " << tauIpInfo.size() << " Tau candidates" ;

	TauImpactParameterInfoCollection::const_iterator iJet;
	for (iJet = tauIpInfo.begin(); iJet != tauIpInfo.end(); iJet++) {

            try{

	      double Rmatch  = 0.1,
		     Rsignal = 0.07,
		     Riso    = 0.4,
		     pT_LT   = 6,
		     pT_min  = 1;
	      double discriminator = iJet->getIsolatedTauTag()->discriminator(Rmatch,Rsignal,Riso,pT_LT,pT_min);

	      const Jet* theJet = iJet->getIsolatedTauTag()->jet().get();
	      LogInfo("TauImpactParameterTest") << "  Candidate jet Et = " << theJet->et() ;
              LogInfo("TauImpactParameterTest") << "    isolation discriminator = "<< discriminator ;

              if(discriminator == 0) continue;
              if(theJet->et() < 0 || theJet->et() > 150) continue;
	      if(fabs(theJet->eta()) > 2.2) continue;

	      const TrackRefVector& tracks = iJet->getIsolatedTauTag()->selectedTracks();

	      double cone  = 0.1;
	      double ptmin = 6;
	      const TrackRef leadingTrack = iJet->getIsolatedTauTag()->leadingSignalTrack(cone,ptmin);

	      if((leadingTrack)->numberOfValidHits() < 8 || leadingTrack->normalizedChi2() > 10) continue;

	      if(!leadingTrack) continue;

              RefVector<TrackCollection>::const_iterator iTrack;

	      std::vector<TrackRef> tauTracks;
  	      for (iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++){

                  LogInfo("TauImpactParameterTest") << "    track pt, eta "  << (*iTrack)->pt()  << " " 
                                                << (*iTrack)->eta() << " " 
                                                << (*iTrack)->d0()  << " "
                                                << (*iTrack)->d0Error()  << " "
                                                << (*iTrack)->dz()  << " "
                                                << (*iTrack)->dzError()  << " "
						<< (*iTrack)->recHitsSize() << " "
                                                << (*iTrack)->normalizedChi2() ;

		  if((*iTrack)->pt() < 1.0) continue;
		  tauTracks.push_back(*iTrack);
	      }

	      if(tauTracks.size() == 1 || tauTracks.size() == 3){
		const reco::TauImpactParameterTrackData* trackData = iJet->getTrackData(leadingTrack);
		if(! trackData == 0){
		    Measurement1D tip = trackData->transverseIp;
		    Measurement1D tzip = trackData->ip3D;

                    LogInfo("TauImpactParameterTest") << "          ip,sip,err " << tip.value()
                         << " "                     << tip.significance()
                         << " "                     << tip.error() ;
                    LogInfo("TauImpactParameterTest") << "       3D ip,sip,err " << tzip.value()
                         << " "                     << tzip.significance()
                         << " "                     << tzip.error() ;

		    h_sip2d_leadingTrack->Fill(tip.significance());
                    h_sip3d_leadingTrack->Fill(tzip.significance());

                    if(tauTracks.size() == 1) {
			h_ip2d_1prong->Fill(10*tip.value()); // 10* = conversion to mm
                        h_ip3d_1prong->Fill(10*tip.value()); // 10* = conversion to mm
		    }
                    if(tauTracks.size() == 3) {
			h_ip2d_3prong_leadingTrack->Fill(10*tip.value()); // 10* = conversion to mm
                        h_ip3d_3prong_leadingTrack->Fill(10*tzip.value()); // 10* = conversion to mm
			for(std::vector<TrackRef>::const_iterator iTr = tauTracks.begin();
			                                     iTr!= tauTracks.end(); iTr++){
			  const reco::TauImpactParameterTrackData* trData = iJet->getTrackData(*iTr);
			  Measurement1D tip = trData->transverseIp;
			  h_ip2d_3prong->Fill(10*tip.value()); // 10* = conversion to mm

                          Measurement1D tzip = trData->ip3D;
                          h_ip3d_3prong->Fill(10*tzip.value()); // 10* = conversion to mm
			}
		    }
		    if(tauTracks.size() == 1 || tauTracks.size() == 3) {
//                    if(tauTracks.size() == 1 && tip.value() < 0.03) { // ipt < 300um like in fig 12 (CMS Note 2006/028)

			ip2d = 10*tip.value();
			ip3d = 10*tzip.value();
			sip2d = tip.significance();
			sip3d = tzip.significance();
			LogDebug("TauImpactParameterTest") << "check tree " << ip2d << " " << sip2d << " " << ip3d << " " << sip3d ;
			t_performance->Fill();
		    }
                }else{
                    LogInfo("TauImpactParameterTest") << "    track data = 0! " ;
		}
	      }


	    }catch ( std::exception & e){
	      LogInfo("TauImpactParameterTest") << "Genexception: " << e.what() ;
	    }
	}
}

void TauImpactParameterTest::beginJob(){
}

void TauImpactParameterTest::endJob(){

  LogInfo("TauImpactParameterTest") << " Events analysed " << nevents ;

	// ip performance plot calculation
	const int N = 50;
	double  x[N],
		y2d[N],
		y3d[N];


        for(int iCut = 0; iCut < N; iCut++){
          double cut = 1./N*iCut;

	  int ip2d_eff = 0;
          int ip3d_eff = 0;

	  int n_entries = t_performance->GetEntries();
	  for(int i = 0; i < n_entries; i++){
		t_performance->GetEntry(i);
		if(ip2d > cut) ip2d_eff++;
		if(ip3d > cut) ip3d_eff++;
	  }

	  double eff2d = 1.*ip2d_eff/n_entries;
          double eff3d = 1.*ip3d_eff/n_entries;

	  x[iCut] = cut;
	  y2d[iCut] = eff2d;
          y3d[iCut] = eff3d;
	}
	// sip performance plot calculation
	const int M = 21;
        double  sx[M],
                sy2d[M],
                sy3d[M];

        for(int iCut = 0; iCut < M; iCut++){
          double cut = iCut;

          int sip2d_eff = 0;
          int sip3d_eff = 0;

          int n_entries = t_performance->GetEntries();
          for(int i = 0; i < n_entries; i++){
                t_performance->GetEntry(i);

                if(sip2d > cut) sip2d_eff++;
                if(sip3d > cut) sip3d_eff++;
          }

          double eff2d = 1.*sip2d_eff/n_entries;
          double eff3d = 1.*sip3d_eff/n_entries;

          sx[iCut] = cut;
          sy2d[iCut] = eff2d;
          sy3d[iCut] = eff3d;
        }

	//


        TCanvas* tauip = new TCanvas("tauip","",500,500);
	tauip->SetFillColor(0);
        tauip->Divide(3,2);


        tauip->cd(1);
	  gPad->SetLogy();
	  h_ip2d_1prong->SetStats(0);
	  LogInfo("TauImpactParameterTest") << "1-prong taus: " << h_ip2d_1prong->GetEntries() ;
	  if(h_ip2d_1prong->GetMaximum() > 0) h_ip2d_1prong->Scale(1/h_ip2d_1prong->GetMaximum());
          h_ip2d_1prong->GetXaxis()->SetTitle("ip_{T} (mm)");
          h_ip2d_1prong->Draw();
	  TLatex* text1 = new TLatex(0.25,1,"1-prong tau tracks");	  
	  text1->Draw();

        tauip->cd(2);
          gPad->SetLogy();
          h_ip2d_3prong_leadingTrack->SetStats(0);
          LogInfo("TauImpactParameterTest") << "3-prong taus: " << h_ip2d_3prong_leadingTrack->GetEntries() ;
          if(h_ip2d_3prong_leadingTrack->GetMaximum() > 0) h_ip2d_3prong_leadingTrack->Scale(1/h_ip2d_3prong_leadingTrack->GetMaximum());
          h_ip2d_3prong_leadingTrack->GetXaxis()->SetTitle("ip_{T} (mm)");
          h_ip2d_3prong_leadingTrack->Draw();
          TLatex* text2 = new TLatex(0.25,1,"3-prong tau tracks");
          text2->Draw();
          TLatex* text3 = new TLatex(0.25,0.7,"leading track ip");
          text3->Draw();

        tauip->cd(3);
          gPad->SetLogy();
	  h_ip2d_3prong->SetStats(0);
          LogInfo("TauImpactParameterTest") << "3-prong taus: " << h_ip2d_3prong->GetEntries() ;
          if(h_ip2d_3prong->GetMaximum() > 0) h_ip2d_3prong->Scale(1/h_ip2d_3prong->GetMaximum());
	  h_ip2d_3prong->GetXaxis()->SetTitle("ip_{T} (mm)");
          h_ip2d_3prong->Draw();
          TLatex* text4 = new TLatex(0.25,1,"3-prong tau tracks");
	  text4->Draw();
          TLatex* text5 = new TLatex(0.25,0.7,"all tracks");
          text5->Draw();

	tauip->cd(4);
          gPad->SetLogy();

          TH2F* frame = new TH2F("frame","",50,0,1,100,0.001,1);
	  frame->SetStats(0);
	  frame->GetXaxis()->SetTitle("ipt cut (mm)");
	  frame->GetYaxis()->SetTitle("efficiency");
	  frame->Draw();
	  TGraph* g_performance2d = new TGraph(N,x,y2d);
	  g_performance2d->SetMarkerStyle(21);
          g_performance2d->SetMarkerSize(0.5);
	  g_performance2d->Draw("PSAME");
          TGraph* g_performance3d = new TGraph(N,x,y3d);
          g_performance3d->SetMarkerStyle(22);
          g_performance3d->SetMarkerSize(0.5);
//          g_performance3d->Draw("PSAME");

        tauip->cd(5);
          gPad->SetLogy();
	  h_sip2d_leadingTrack->SetStats(0);
	  if(h_sip2d_leadingTrack->GetMaximum() > 0) h_sip2d_leadingTrack->Scale(1/h_sip2d_leadingTrack->GetMaximum());
          h_sip2d_leadingTrack->GetXaxis()->SetTitle("#sigma_{ipt}");
	  h_sip2d_leadingTrack->Draw();

        tauip->cd(6);
          gPad->SetLogy();

          TH2F* frame_s = new TH2F("frame_s","",10,0,M,10,0.01,1);
          frame_s->SetStats(0);
          frame_s->GetXaxis()->SetTitle("#sigma_{ipt} cut");
          frame_s->GetYaxis()->SetTitle("efficiency");
          frame_s->Draw();
          TGraph* g_performance_s2d = new TGraph(M,sx,sy2d);
          g_performance_s2d->SetMarkerStyle(21);
          g_performance_s2d->SetMarkerSize(0.5);
          g_performance_s2d->Draw("PSAME");
	

        tauip->Print("tauip.C");

///////////////////////////////////////////////////////////////////////
	{
        TCanvas* tauip3D = new TCanvas("tauip3D","",500,500);
        tauip3D->SetFillColor(0);
        tauip3D->Divide(3,2);


        tauip3D->cd(1);
          gPad->SetLogy();
          h_ip3d_1prong->SetStats(0);
          LogInfo("TauImpactParameterTest") << "1-prong taus: " << h_ip3d_1prong->GetEntries() ;
          if(h_ip3d_1prong->GetMaximum() > 0) h_ip3d_1prong->Scale(1/h_ip3d_1prong->GetMaximum());
          h_ip3d_1prong->GetXaxis()->SetTitle("ip_{3D} (mm)");
          h_ip3d_1prong->Draw();
          TLatex* text1 = new TLatex(0.25,1,"1-prong tau tracks");
          text1->Draw();

        tauip3D->cd(2);
          gPad->SetLogy();
          h_ip3d_3prong_leadingTrack->SetStats(0);
          LogInfo("TauImpactParameterTest") << "3-prong taus: " << h_ip3d_3prong_leadingTrack->GetEntries() ;
          if(h_ip3d_3prong_leadingTrack->GetMaximum() > 0) h_ip3d_3prong_leadingTrack->Scale(1/h_ip3d_3prong_leadingTrack->GetMaximum());
          h_ip3d_3prong_leadingTrack->GetXaxis()->SetTitle("ip_{3D} (mm)");
          h_ip3d_3prong_leadingTrack->Draw();
          TLatex* text2 = new TLatex(0.25,1,"3-prong tau tracks");
          text2->Draw();
          TLatex* text3 = new TLatex(0.25,0.7,"leading track ip");
          text3->Draw();

        tauip3D->cd(3);
          gPad->SetLogy();
          h_ip3d_3prong->SetStats(0);
          LogInfo("TauImpactParameterTest") << "3-prong taus: " << h_ip3d_3prong->GetEntries() ;
          if(h_ip3d_3prong->GetMaximum() > 0) h_ip3d_3prong->Scale(1/h_ip3d_3prong->GetMaximum());
          h_ip3d_3prong->GetXaxis()->SetTitle("ip_{3D} (mm)");
          h_ip3d_3prong->Draw();
          TLatex* text4 = new TLatex(0.25,1,"3-prong tau tracks");
          text4->Draw();
          TLatex* text5 = new TLatex(0.25,0.7,"all tracks");
          text5->Draw();

        tauip3D->cd(4);
          gPad->SetLogy();

          TH2F* frame3D = new TH2F("frame3D","",50,0,1,100,0.001,1);
          frame3D->SetStats(0);
          frame3D->GetXaxis()->SetTitle("ip3D cut (mm)");
          frame3D->GetYaxis()->SetTitle("efficiency");
          frame3D->Draw();
          TGraph* g_performance3d = new TGraph(N,x,y3d);
          g_performance3d->SetMarkerStyle(21);
          g_performance3d->SetMarkerSize(0.5);
          g_performance3d->Draw("PSAME");

        tauip3D->cd(5);
          gPad->SetLogy();
          h_sip3d_leadingTrack->SetStats(0);
          if(h_sip3d_leadingTrack->GetMaximum() > 0) h_sip3d_leadingTrack->Scale(1/h_sip3d_leadingTrack->GetMaximum());
          h_sip3d_leadingTrack->GetXaxis()->SetTitle("#sigma_{ip3D}");
          h_sip3d_leadingTrack->Draw();

        tauip3D->cd(6);
          gPad->SetLogy();

          TH2F* frame3D_s = new TH2F("frame3D_s","",10,0,M,10,0.01,1);
          frame3D_s->SetStats(0);
          frame3D_s->GetXaxis()->SetTitle("sip3D cut");
          frame3D_s->GetYaxis()->SetTitle("efficiency");
          frame3D_s->Draw();
          TGraph* g_performance_s3d = new TGraph(M,sx,sy3d);
          g_performance_s3d->SetMarkerStyle(21);
          g_performance_s3d->SetMarkerSize(0.5);
          g_performance_s3d->Draw("PSAME");

        tauip3D->Print("tauip3D.C");
	}
}

//define this as a plug-in

DEFINE_FWK_MODULE(TauImpactParameterTest);
