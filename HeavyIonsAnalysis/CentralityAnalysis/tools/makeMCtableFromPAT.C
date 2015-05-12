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

void makeMCtableFromPAT(int nbins = 10, const string label = "hfhit", const char * tag = "HFhitBins"){

   // Retrieving data
  int nFiles = 1;
  int maxEvents = -200;
  vector<string> infiles;
  
  infiles.push_back("dcache:/pnfs/cmsaf.mit.edu/t2bat/cms/store/himc/Fall10/AMPT_Default_MinBias_2760GeV/GEN-SIM-RECO/MC_38Y_V12-v1/0002/F8E134B2-77D6-DF11-BFE1-001B243DE10F.root");
  
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

  bool binNpart = label.compare("Npart") == 0;
  bool binNcoll = label.compare("Ncoll") == 0;
  bool binNhard = label.compare("Nhard") == 0;
  bool binB = label.compare("b") == 0;
  bool binHF = label.compare("HFtowers") == 0;
  bool binHFhit = label.compare("HFhits") == 0;
  bool binEB = label.compare("EB") == 0;
  bool binEE = label.compare("EE") == 0;
  bool binETMR = label.compare("ETMR") == 0;
  bool binNpix = label.compare("PixelHits") == 0;
  bool binNtrks = label.compare("Ntracks") == 0;

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

    double hf = cent->EtHFtowerSum();
    double hfhit = cent->EtHFhitSum();
    double eb = cent->EtEBSum();
    double eep = cent->EtEESumPlus();
    double eem = cent->EtEESumMinus();
    double ee = cent->EtEESum();
    double etmr = cent->EtMidRapiditySum();
    double npix = cent->multiplicityPixel();
    double ntrks = cent->Ntracks();

    double parameter = 0;
    if(binNpart) parameter = npart;
    if(binNcoll) parameter = ncoll;
    if(binNhard) parameter = nhard;
    if(binB) parameter = b;
    if(binHF) parameter = hf;
    if(binHFhit) parameter = hfhit;
    if(binEB) parameter = eb;
    if(binEE) parameter = ee;
    if(binETMR) parameter = etmr;
    if(binNpix) parameter = npix;
    if(binNtrks) parameter = ntrks;

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

     double hf = cent->EtHFtowerSum();
     double hfhit = cent->EtHFhitSum();
     double eb = cent->EtEBSum();
     double eep = cent->EtEESumPlus();
     double eem = cent->EtEESumMinus();
     double ee = cent->EtEESum();
     double etmr = cent->EtMidRapiditySum();
     double npix = cent->multiplicityPixel();
     double ntrks = cent->Ntracks();

     double parameter = 0;
     if(binNpart) parameter = npart;
     if(binNcoll) parameter = ncoll;
     if(binNhard) parameter = nhard;
     if(binB) parameter = b;
     if(binHF) parameter = hf;
     if(binHFhit) parameter = hfhit;
     if(binEB) parameter = eb;
     if(binEE) parameter = ee;
     if(binETMR) parameter = etmr;
     if(binNpix) parameter = npix;
     if(binNtrks) parameter = ntrks;
    
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
     int ii = nbins-i;
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

   for(int i = 1; i < nBins+1; ++i){
      int bin = nBins - i;
      TH1D* h = hCorr->ProjectionY(Form("%s_bin%d",hCorr->GetName(),bin),i,i);

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




