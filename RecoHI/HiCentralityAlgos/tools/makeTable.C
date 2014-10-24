#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <TF1.h>
#include <TH2D.h>
#include <TH1D.h>
#include <TNtuple.h>
#include <TChain.h>
#include <TFile.h>
#include <TSystem.h>

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#endif

void fitSlices(TH2*, TF1*);

static bool ZS = false;
static bool useFits = false;
static bool onlySaveTable = false;
static const int nbinsMax = 40;

using namespace std;
bool descend(float i,float j) { return (i<j); }

void makeTable(int nbins = 40, const string label = "HFhits", const char * tag = "Preliminary_NoEffCor_AMPT_d1107", const char* dataset = "DATA"){

  bool DATA = false;
  bool SIM = true;
  bool MC = false;
  double EFF = 1;
  double MXS = 1. - EFF;

   // Retrieving data
  int maxEvents = -200;
  vector<int> runnums;
  
  //  const char* infileName = Form("/net/hisrv0001/home/yetkin/hidsk0001/analysis/prod/%s_RECO_391/test.root",dataset);
  const char* infileName = Form("/net/hisrv0001/home/yetkin/hidsk0001/centrality/prod/%s/test.root",dataset);

  //  TFile* infile = new TFile(infileName,"read");
  TChain* t = new TChain("HltTree");
  //  TChain* t = new TChain("hltanalysis/HltTree");

  t->Add(infileName);

  // Creating output table
  TFile* outFile = new TFile("tables_d1108.root","update");
   TDirectory* dir = outFile->mkdir(tag);
   dir->cd();
   TNtuple* nt = new TNtuple("nt","","hf:bin:b:npart:ncoll:nhard");
   CentralityBins* bins = new CentralityBins("noname","Test tag", nbins);
   bins->table_.reserve(nbins);

  TH1D::SetDefaultSumw2();

  int runMC = 1;
  TFile * inputMCfile;
  CentralityBins* inputMCtable;
  
  if(DATA){
    inputMCfile = new TFile("tables_d1103.root","read");
    inputMCtable = (CentralityBins*)inputMCfile->Get("CentralityTable_HFhits40_AMPT2760GeV_v1_mc_MC_38Y_V12/run1");
  }

  // Setting up variables & branches
  double binboundaries[nbinsMax+1];
  vector<float> values;

  float b,npart,ncoll,nhard,hf,hfhit,eb,ee,etmr,parameter;
  int npix,ntrks;
  //  TTree* t = (TTree*)infile->Get("HltTree");
  int run;

  if(SIM){
    t->SetBranchAddress("b",&b);
    t->SetBranchAddress("Npart",&npart);
    t->SetBranchAddress("Ncoll",&ncoll);
    t->SetBranchAddress("Nhard",&nhard);
  }

  t->SetBranchAddress("hiHFhit",&hfhit);
  t->SetBranchAddress("hiHF",&hf);
  t->SetBranchAddress("hiEB",&eb);
  t->SetBranchAddress("hiEE",&ee);
  t->SetBranchAddress("hiET",&etmr);
  t->SetBranchAddress("hiNpix",&npix);
  t->SetBranchAddress("hiNtracks",&ntrks);
  t->SetBranchAddress("Run",&run);

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
  unsigned int events=t->GetEntries();
  for(unsigned int iev = 0; iev < events && (maxEvents < 0 || iev< maxEvents); ++iev){
    if( iev % 100 == 0 ) cout<<"Processing event : "<<iev<<endl;
    t->GetEntry(iev);

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
  double dev = events;
  for(int i = 0; i< nbins; ++i){
     // Find the boundary 
    int entry = (int)(i*(dev/nbins));
    binboundaries[i] = values[entry];

     cout<<" "<<binboundaries[i];
     if(i < nbins - 1) cout<<",";
     else cout<<")"<<endl;
  }

  cout<<"-------------------------------------"<<endl;

  if(!DATA){

  // Determining Glauber results in various bins
  dir->cd();
  TH2D* hNpart = new TH2D("hNpart","",nbins,binboundaries,500,0,500);
  TH2D* hNcoll = new TH2D("hNcoll","",nbins,binboundaries,2000,0,2000);
  TH2D* hNhard = new TH2D("hNhard","",nbins,binboundaries,250,0,250);
  TH2D* hb = new TH2D("hb","",nbins,binboundaries,300,0,30);

  for(unsigned int iev = 0; iev < events && (maxEvents < 0 || iev< maxEvents); ++iev){
     if( iev % 100 == 0 ) cout<<"Processing event : "<<iev<<endl;
     t->GetEntry(iev);
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
     bins->table_[i].bin_edge = binboundaries[ii-1];

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
 
  }else{
    cout<<"-------------------------------------"<<endl;
    cout<<"# Bin NpartMean NpartSigma NcollMean NcollSigma bMean bSigma BinEdge"<<endl;

    // Enter values in table
    for(int i = 0; i < nbins; ++i){
      int ii = nbins-i;
      bins->table_[i].n_part_mean = inputMCtable->NpartMeanOfBin(i);
      bins->table_[i].n_part_var = inputMCtable->NpartSigmaOfBin(i);
      bins->table_[i].n_coll_mean = inputMCtable->NcollMeanOfBin(i);
      bins->table_[i].n_coll_var = inputMCtable->NcollSigmaOfBin(i);
      bins->table_[i].b_mean = inputMCtable->bMeanOfBin(i);
      bins->table_[i].b_var = inputMCtable->bSigmaOfBin(i);
      bins->table_[i].n_hard_mean = inputMCtable->NhardMeanOfBin(i);
      bins->table_[i].n_hard_var = inputMCtable->NhardSigmaOfBin(i);
      bins->table_[i].bin_edge = binboundaries[ii-1];

      cout<<i<<" "
	  <<bins->table_[i].n_part_mean<<" "
          <<bins->table_[i].n_part_var<<" "
          <<bins->table_[i].n_coll_mean<<" "
          <<bins->table_[i].n_coll_var<<" "
          <<bins->table_[i].b_mean<<" "
          <<bins->table_[i].b_var<<" "
          <<bins->table_[i].n_hard_mean<<" "
          <<bins->table_[i].n_hard_var<<" "
          <<bins->table_[i].bin_edge<<" "<<endl;

    }
    cout<<"-------------------------------------"<<endl;

  }


  outFile->cd(); 
  dir->cd();

  bins->SetName(Form("run%d",1));
  bins->Write();
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





