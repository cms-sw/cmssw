#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <TF1.h>
#include <TH2D.h>
#include <TH1D.h>
#include <TNtuple.h>
#include <TFile.h>
#include <TSystem.h>

#if !defined(__CINT__) && !defined(__MAKECINT__)


#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

#endif

void fitSlices(TH2*, TF1*);

static bool ZS = false;
static bool useFits = false;
static bool onlySaveTable = false;
static const int nbinsMax = 40;

using namespace std;
bool descend(float i,float j) { return (i<j); }

void makeMCCentralityTable(int nbins = 10, const string label = "hf", const char * tag = "HFhitBins"){

   // Retrieving data
  int nFiles = 1;
  int maxEvents = -200;
  vector<string> infiles;

  if(ZS){
     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_ZS_01/AMPT_MB_ZS_0711_runs1to50.root");
     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_ZS_01/AMPT_MB_ZS_0711_runs51to100.root");
     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_ZS_01/AMPT_MB_ZS_0711_runs101to150.root");
  }else{
     //     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_NZS_02/AMPT_MB_ZS_0711_runs1to20.root");
     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_NZS_02/AMPT_MB_ZS_0711_runs21to40.root");
     //     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_NZS_02/AMPT_MB_ZS_0711_runs41to60.root");
     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_NZS_02/AMPT_MB_ZS_0711_runs61to80.root");
     //     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_NZS_02/AMPT_MB_ZS_0711_runs81to100.root");
     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_NZS_02/AMPT_MB_ZS_0711_runs101to120.root");
     //     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_NZS_02/AMPT_MB_ZS_0711_runs121to140.root");
     infiles.push_back("/net/hisrv0001/home/yetkin/hidsk0001/aod/JulyExercise/AMPT_NZS_02/AMPT_MB_ZS_0711_runs141to160.root");
  }

  fwlite::ChainEvent event(infiles);

    //    TFile* infile = new TFile("/net/hisrv0001/home/yetkin/pstore02/ana/Hydjet_MinBias_2760GeV_d20100305/Hydjet_MinBias_2760GeV_runs1to1000.root");
  //  TFile* infile = new TFile("/net/hisrv0001/home/yetkin/pstore02/ana/Hydjet_MinBias_4TeV_d20100305/Hydjet_MinBias_4TeV_runs1to500.root");
    //  fwlite::Event event(infile);
  vector<int> runnums;

  // Creating output table
  TFile* outFile = new TFile("tables10binsHalfEvents.root","update");
   TDirectory* dir = outFile->mkdir(tag);
   dir->cd();
   TNtuple* nt = new TNtuple("nt","","hf:bin:b:npart:ncoll:nhard");

  TH1D::SetDefaultSumw2();
  CentralityBins* bins = new CentralityBins("noname","Test tag", nbins);
  bins->table_.reserve(nbins);

  // Setting up variables & branches
  double binboundaries[nbinsMax+1];
  vector<float> values;

  // Determining bins of cross section
  // loop over events
  unsigned int events=0;
  for(event.toBegin(); !event.atEnd() && (maxEvents < 0 || events< maxEvents); ++event, ++events){
     edm::EventBase const & ev = event;
    if( events % 100 == 0 ) cout<<"Processing event : "<<events<<endl;
    edm::Handle<edm::GenHIEvent> mc;
    ev.getByLabel(edm::InputTag("heavyIon"),mc);
    edm::Handle<reco::Centrality> cent;
    ev.getByLabel(edm::InputTag("hiCentrality"),cent);

    double b = mc->b();
    double npart = mc->Npart();
    double ncoll = mc->Ncoll();
    double nhard = mc->Nhard();

    double hf = cent->EtHFhitSum();
    double hftp = cent->EtHFtowerSumPlus();
    double hftm = cent->EtHFtowerSumMinus();
    double eb = cent->EtEBSum();
    double eep = cent->EtEESumPlus();
    double eem = cent->EtEESumMinus();
    double etmr = cent->EtMidRapiditySum();
    double npix = cent->multiplicityPixel();

    double parameter = 0;
    if(label.compare("npart") == 0) parameter = npart;
    if(label.compare("ncoll") == 0) parameter = ncoll;
    if(label.compare("nhard") == 0) parameter = nhard;
    if(label.compare("b") == 0) parameter = b;
    if(label.compare("hf") == 0) parameter = hf;
    if(label.compare("hft") == 0) parameter = hftp + hftm;
    if(label.compare("eb") == 0) parameter = eb;
    if(label.compare("ee") == 0) parameter = eep+eem;
    if(label.compare("etmr") == 0) parameter = etmr;
    if(label.compare("npix") == 0) parameter = npix;

    values.push_back(parameter);
    
    int run = event.id().run();
    if(runnums.size() == 0 || runnums[runnums.size()-1] != run) runnums.push_back(run);
  }
  
  if(label.compare("b") == 0) sort(values.begin(),values.end(),descend);
  else sort(values.begin(),values.end());

  double max = values[events-1];
  binboundaries[nbins] = max;

  cout<<"-------------------------------------"<<endl;
  cout<<label.data()<<" based cuts are : "<<endl;
  cout<<"(";

  int bin = 0;
  for(int i = 0; i< nbins; ++i){
     // Find the boundary 
     int entry = (int)(i*(events/nbins));
     binboundaries[i] = values[entry];

     cout<<" "<<binboundaries[i];
     if(i < nbins - 1) cout<<",";
     else cout<<")"<<endl;
  }

  cout<<"-------------------------------------"<<endl;

  // Determining Glauber results in various bins
  dir->cd();
  TH2D* hNpart = new TH2D("hNpart","",nbins,binboundaries,500,0,500);
  TH2D* hNcoll = new TH2D("hNcoll","",nbins,binboundaries,2000,0,2000);
  TH2D* hNhard = new TH2D("hNhard","",nbins,binboundaries,250,0,250);
  TH2D* hb = new TH2D("hb","",nbins,binboundaries,300,0,30);

  for(event.toBegin(); !event.atEnd(); ++event){
     edm::EventBase const & ev = event;
     edm::Handle<edm::GenHIEvent> mc;
     ev.getByLabel(edm::InputTag("heavyIon"),mc);
     edm::Handle<reco::Centrality> cent;
     ev.getByLabel(edm::InputTag("hiCentrality"),cent);

     double b = mc->b();
     double npart = mc->Npart();
     double ncoll = mc->Ncoll();
     double nhard = mc->Nhard();

     double hf = cent->EtHFhitSum();
     double hftp = cent->EtHFtowerSumPlus();
     double hftm = cent->EtHFtowerSumMinus();
     double eb = cent->EtEBSum();
     double eep = cent->EtEESumPlus();
     double eem = cent->EtEESumMinus();
     double etmr = cent->EtMidRapiditySum();
     double npix = cent->multiplicityPixel();

     double parameter = 0;

     if(label.compare("npart") == 0) parameter = npart;
     if(label.compare("ncoll") == 0) parameter = ncoll;
     if(label.compare("nhard") == 0) parameter = nhard;
     if(label.compare("b") == 0) parameter = b;
     if(label.compare("hf") == 0) parameter = hf;
     if(label.compare("hft") == 0) parameter = hftp + hftm;
     if(label.compare("eb") == 0) parameter = eb;
     if(label.compare("ee") == 0) parameter = eep+eem;
     if(label.compare("etmr") == 0) parameter = etmr;
     if(label.compare("npix") == 0) parameter = npix;
    
     hNpart->Fill(parameter,npart);
     hNcoll->Fill(parameter,ncoll);
     hNhard->Fill(parameter,nhard);
     hb->Fill(parameter,b);
     int bin = hNpart->GetXaxis()->FindBin(parameter) - 1;
     if(bin < 0) bin = 0;
     if(bin >= nbins) bin = nbins - 1;
     nt->Fill(hf,bin,b,npart,ncoll,nhard);
  }

  // Fitting Glauber distributions in bins to get mean and sigma values

  dir->cd();
  TF1* fGaus = new TF1("fb","gaus(0)",0,2); 
  fGaus->SetParameter(0,1);
  fGaus->SetParameter(1,0.04);
  fGaus->SetParameter(2,0.02); 
  
  fitSlices(hNpart,fGaus);
  fitSlices(hNcoll,fGaus);
  fitSlices(hNhard,fGaus);
  fitSlices(hb,fGaus);

  TH1D* hNpartMean = (TH1D*)gDirectory->Get("hNpart_1");
  TH1D* hNpartSigma = (TH1D*)gDirectory->Get("hNpart_2");
  TH1D* hNcollMean = (TH1D*)gDirectory->Get("hNcoll_1");
  TH1D* hNcollSigma = (TH1D*)gDirectory->Get("hNcoll_2");
  TH1D* hNhardMean = (TH1D*)gDirectory->Get("hNhard_1");
  TH1D* hNhardSigma = (TH1D*)gDirectory->Get("hNhard_2");
  TH1D* hbMean = (TH1D*)gDirectory->Get("hb_1");
  TH1D* hbSigma = (TH1D*)gDirectory->Get("hb_2");

  cout<<"-------------------------------------"<<endl;
  cout<<"# Bin NpartMean NpartSigma NcollMean NcollSigma bMean bSigma BinEdge"<<endl;


  // Enter values in table
  for(int i = 0; i < nbins; ++i){
     int ii = nbins-i-1;
     bins->table_[i].n_part_mean = hNpartMean->GetBinContent(ii);
     bins->table_[i].n_part_var = hNpartSigma->GetBinContent(ii);
     bins->table_[i].n_coll_mean = hNcollMean->GetBinContent(ii);
     bins->table_[i].n_coll_var = hNcollSigma->GetBinContent(ii);
     bins->table_[i].b_mean = hbMean->GetBinContent(ii);
     bins->table_[i].b_var = hbSigma->GetBinContent(ii);
     bins->table_[i].n_hard_mean = hNhardMean->GetBinContent(ii);
     bins->table_[i].n_hard_var = hNhardSigma->GetBinContent(ii);
     bins->table_[i].bin_edge = binboundaries[ii];

     cout<<i<<" "
	 <<hNpartMean->GetBinContent(ii)<<" "
	 <<hNpartSigma->GetBinContent(ii)<<" "
	 <<hNcollMean->GetBinContent(ii)<<" "
	 <<hNcollSigma->GetBinContent(ii)<<" "
	 <<hbMean->GetBinContent(ii)<<" "
	 <<hbSigma->GetBinContent(ii)<<" "
	 <<binboundaries[ii]<<" "
	 <<endl;
  }
  cout<<"-------------------------------------"<<endl;

  // Save the table in output file
  if(onlySaveTable){

     hNpart->Delete();
     hNpartMean->Delete();
     hNpartSigma->Delete();
     hNcoll->Delete();
     hNcollMean->Delete();
     hNcollSigma->Delete();
     hNhard->Delete();
     hNhardMean->Delete();
     hNhardSigma->Delete();
     hb->Delete();
     hbMean->Delete();
     hbSigma->Delete();
  }
  
  //  for(int i = 0; i < runnums.size(); ++i){
  CentralityBins* binsForRun = (CentralityBins*) bins->Clone();
  binsForRun->SetName(Form("run%d",1));
  //  binsForRun->SetName(Form("run%d",runnums[i]));
  binsForRun->Write();
     //  }
  nt->Write();  
  bins->Delete();
  outFile->Write();
  
}

void fitSlices(TH2* hCorr, TF1* func){

   int nBins = hCorr->GetNbinsX();

   TH1D* hMean = new TH1D(Form("%s_1",hCorr->GetName()),"",nBins,hCorr->GetXaxis()->GetXmin(),hCorr->GetXaxis()->GetXmax());
   TH1D* hSigma = new TH1D(Form("%s_2",hCorr->GetName()),"",nBins,hCorr->GetXaxis()->GetXmin(),hCorr->GetXaxis()->GetXmax());

   for(int i = 0; i < nBins; ++i){
      TH1D* h = hCorr->ProjectionY(Form("%s_bin%d",hCorr->GetName(),i),i,i+1);

	 func->SetParameter(0,h->GetMaximum());
	 func->SetParameter(1,h->GetMean());
	 func->SetParameter(2,h->GetRMS());

	 if(useFits) h->Fit(func);

	 hMean->SetBinContent(i,func->GetParameter(1));
	 hMean->SetBinError(i,func->GetParError(1));
	 hSigma->SetBinContent(i,func->GetParameter(2));
	 hSigma->SetBinError(i,func->GetParError(2));

      if(onlySaveTable){
	 h->Delete();
      }
   }
}





