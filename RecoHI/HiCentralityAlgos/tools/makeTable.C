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

void makeTable(int nbins = 40, const string label = "HFhits", const char * tag = "Preliminary_Eff97_AMPT_run150590_d1114v0", const char* dataset = "DATA",const char* glauber = "HydjetBass"){

  bool DATA = true;
  bool SIM = false;
  bool MC = false;
  double EFF = 0.97;
  double MXS = 1. - EFF;

  const char* bit0 = "HLT_HIMinBiasBSC_Core";
  const char* bit1 = "HLT_HIClusterVertexCompatibility";
  
  string beamhalobits[100] = {
    "L1Tech_BSC_halo_beam2_inner.v0",
    "L1Tech_BSC_halo_beam2_outer.v0",
    "L1Tech_BSC_halo_beam1_inner.v0",
    "L1Tech_BSC_halo_beam1_outer.v0"
  };

   // Retrieving data
  int maxEvents = -200;
  vector<int> runnums;
  
  string infileNames[500] = {
    Form("/d101/frankma/data/HIAllPhysics/HR10AllPR2/r150590v1/*.root",dataset)
  };

  int nFiles = 1;
  TChain* t = new TChain("hltanalysis/HltTree");

  for(int i = 0; i < nFiles; ++i){
    t->Add(infileNames[i].data());
  }

  // Creating output table
  TFile* outFile = new TFile("tables_Run150590_AMPTOrgan_d1114.root","update");
   TDirectory* dir = outFile->mkdir(tag);
   dir->cd();
   TNtuple* nt = new TNtuple("nt","","hf:bin:b:npart:ncoll:nhard");
   CentralityBins* bins = new CentralityBins("noname","Test tag", nbins);
   bins->table_.reserve(nbins);

  TH1D::SetDefaultSumw2();

  int runMC = 1;
  TFile * inputMCfile;
  CentralityBins* inputMCtable;

  TH1* hEff;
  if(DATA){
    if(0){
      inputMCfile = new TFile("GuitarPiano_Glauber.root","read");
      inputMCtable = (CentralityBins*)inputMCfile->Get("CentralityTable_HFhits40_AMPT_Piano2760GeV_v3_mc_MC_39Y_V4/run1");
      hEff = (TH1*)inputMCfile->Get(Form("hEff"));
    }else{
      inputMCfile = new TFile("BassOrgan_Glauber.root","read");
      inputMCtable = (CentralityBins*)inputMCfile->Get("CentralityTable_HFhits40_AMPT2760GeV_v1_mc_MC_38Y_V12/run1");
    }
  }

  // Setting up variables & branches
  double binboundaries[nbinsMax+1];
  vector<float> values;

  float b,npart,ncoll,nhard,hf,hfhit,eb,ee,etmr,parameter;
  int npix,ntrks, vtxNtrk;
  int run;
  int trig[20];
  int reject[20];

  t->SetBranchAddress(bit0,&(trig[0]));
  t->SetBranchAddress(bit1,&(trig[1]));
  t->SetBranchAddress(beamhalobits[0].data(),&(reject[0]));
  t->SetBranchAddress(beamhalobits[1].data(),&(reject[1]));
  t->SetBranchAddress(beamhalobits[2].data(),&(reject[2]));
  t->SetBranchAddress(beamhalobits[3].data(),&(reject[3]));
  
  t->SetBranchAddress("recoVrtNtrk",&vtxNtrk);
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
  bool binETMR = label.compare("ETmidRapidity") == 0;
  bool binNpix = label.compare("PixelHits") == 0;
  bool binNtrks = label.compare("Ntracks") == 0;

  // Determining bins of cross section
  // loop over events
  double dev = 0;
  double xsec = 0;

  unsigned int events=t->GetEntries();
  for(unsigned int iev = 0; iev < events && (maxEvents < 0 || iev< maxEvents); ++iev){
    if( iev % 5000 == 0 ) cout<<"Processing event : "<<iev<<endl;
    t->GetEntry(iev);

    bool validVtx = vtxNtrk > 1;
    bool selectEvent = trig[0] && trig[1]  // Min Bias, Vtx compatible
      && 
      !(reject[0] || reject[1] || reject[2] || reject[3]) // Not halo
      && 
      validVtx;

    if(!selectEvent) continue;

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
    //    cout<<"before runnum"<<endl;

    values.push_back(parameter);
    if(runnums.size() == 0 || runnums[runnums.size()-1] != run) runnums.push_back(run);
    //    cout<<"runnum"<<endl;

    dev += 1;
    //    if(0) xsec += parameter / hEff->GetBinContent(hEff->FindBin(val));

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
    int entry = (int)(i*(dev/EFF/nbins) - dev*MXS/EFF);
    binboundaries[i] = values[entry];

     cout<<" "<<binboundaries[i];
     if(i < nbins - 1) cout<<",";
     else cout<<")"<<endl;

  }

  if(0){ // Alternatively, using Npart dependent efficiency
    double integral = 0;
    int currentbin = 0;
    for(int iv = 0; iv < dev && currentbin < nbins; ++iv){
      double val = values[iv];  
      integral += val / hEff->GetBinContent(hEff->FindBin(val));
      if(integral > (int)(currentbin*(xsec/nbins))){
	binboundaries[currentbin] = val;
	currentbin++;
      }
    }
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
     bool validVtx = vtxNtrk > 1;
     bool selectEvent = trig[0] && trig[1]  // Min Bias, Vtx compatible
      &&
       !(reject[0] || reject[1] || reject[2] || reject[3]) // Not halo
      &&
       validVtx;

     if(!selectEvent) continue;

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
  cout<<"a"<<endl;

  dir->cd();
  cout<<"b"<<endl;
  TF1* fGaus = new TF1("fb","gaus(0)",0,2); 
  fGaus->SetParameter(0,1);
  fGaus->SetParameter(1,0.04);
  fGaus->SetParameter(2,0.02); 
  cout<<"c"<<endl;
  
  fitSlices(hNpart,fGaus);
  cout<<"d"<<endl;
  fitSlices(hNcoll,fGaus);
  fitSlices(hNhard,fGaus);
  fitSlices(hb,fGaus);
  cout<<"e"<<endl;

  TH1D* hNpartMean = (TH1D*)gDirectory->Get("hNpart_1");
  TH1D* hNpartSigma = (TH1D*)gDirectory->Get("hNpart_2");
  TH1D* hNcollMean = (TH1D*)gDirectory->Get("hNcoll_1");
  TH1D* hNcollSigma = (TH1D*)gDirectory->Get("hNcoll_2");
  TH1D* hNhardMean = (TH1D*)gDirectory->Get("hNhard_1");
  TH1D* hNhardSigma = (TH1D*)gDirectory->Get("hNhard_2");
  TH1D* hbMean = (TH1D*)gDirectory->Get("hb_1");
  TH1D* hbSigma = (TH1D*)gDirectory->Get("hb_2");
  cout<<"f"<<endl;

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
    cout<<"x"<<endl;

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
     cout<<"y"<<endl;
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

  cout<<"ax"<<endl;

  outFile->cd(); 
  cout<<"axx"<<endl;

  dir->cd();
  cout<<"bx"<<endl;

  bins->SetName(Form("run%d",1));
  cout<<"cx"<<endl;
  bins->Write();
  cout<<"dx"<<endl;
  nt->Write();  
  cout<<"ex"<<endl;
  bins->Delete();
  cout<<"fx"<<endl;
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





