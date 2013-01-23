#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <TF1.h>
#include <TH2D.h>
#include <TH1D.h>
#include <TNtuple.h>
#include <TChain.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TSystem.h>

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "../../../DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#endif

using namespace std;

bool descend(float i,float j) { return (i<j); }

void fitSlices(TH2* hCorr, TF1* func){

   int nBins = hCorr->GetNbinsX();

   TH1D* hMean = new TH1D(Form("%s_1",hCorr->GetName()),"",nBins,hCorr->GetXaxis()->GetXmin(),hCorr->GetXaxis()->GetXmax());
   TH1D* hSigma = new TH1D(Form("%s_2",hCorr->GetName()),"",nBins,hCorr->GetXaxis()->GetXmin(),hCorr->GetXaxis()->GetXmax());

   for(int i = 1; i < nBins+1; i++){
      int bin = nBins - i;
      TH1D* h = hCorr->ProjectionY(Form("%s_bin%d",hCorr->GetName(),bin),i,i);

      func->SetParameter(0,h->GetMaximum());
      func->SetParameter(1,h->GetMean());
      func->SetParameter(2,h->GetRMS());

      h->Fit(func,"Q");

      /*hMean->SetBinContent(i,func->GetParameter(1));
      hMean->SetBinError(i,func->GetParError(1));
      hSigma->SetBinContent(i,func->GetParameter(2));
      hSigma->SetBinError(i,func->GetParError(2));*/

      hMean->SetBinContent(i,h->GetMean());
      hMean->SetBinError(i,func->GetParError(1)); //errors are not used later for the actual table
      hSigma->SetBinContent(i,h->GetRMS());
      hSigma->SetBinError(i,func->GetParError(2));

   }
}

//------------------------------------------------------------------------

void makeTable2(int nbins = 100, const string label = "HFtowersPlusTrunc", const char * tag = "CentralityTable_HFplus100_PA2012B_v538x01_offline", bool isMC = false, int runNum = 1) {

 TH1D::SetDefaultSumw2();

 //Intput files with HiTrees
 const int nTrees = 1;
 //string inFileNames[nTrees] = {"/tmp/azsigmon/HiForest_pPb_Hijing_NEWFIX_v2.root"};
 //string inFileNames[nTrees] = {"/tmp/azsigmon/HiForest_pPb_Epos_336800.root"};
 string inFileNames[nTrees] = {"/tmp/azsigmon/PA2013_HiForest_Express_r0_pilot_minbias_v0.root"};
 TChain * t = new TChain("hiEvtAnalyzer/HiTree");
 for (int i = 0; i<nTrees; i++) {
    t->Add(inFileNames[i].data());
 }

 //Output files and tables
 TFile * outFile = new TFile("out/datatables_Glauber2012B_d20130121_v5.root","recreate");
 //TFile * outFile = new TFile("out/tables_Ampt_d20121115_v3.root","update");
 //TFile * outFile = new TFile("out/tables_Epos_d20121115_v3.root","update");
 //TFile * outFile = new TFile("out/tables_Hijing_d20130119_v4.root","update");
 TDirectory* dir = outFile->mkdir(tag);
 dir->cd();
 TNtuple * nt = new TNtuple("nt","","value:bin:b:npart:ncoll:nhard");
 CentralityBins * bins = new CentralityBins(Form("run%d",runNum), tag, nbins);
 bins->table_.reserve(nbins);

 ofstream txtfile("out/output.txt");
 txtfile << "First input tree: " << inFileNames[0].data() << endl;

 //For data extra inputfile with Glauber centrality table and efficiency file
 TFile * effFile;
 TH1F * hEff; 
 TFile * inputMCfile;
 CentralityBins* inputMCtable;
 if(!isMC){
   //effFile = new TFile("out/efficiencies_Ampt.root","read");
   //effFile = new TFile("out/efficiencies_Hijing.root","read");
   effFile = new TFile("out/efficiencies_EposLHC_v2.root","read");
   hEff = (TH1F*)effFile->Get(Form("%s/hEff",label.data()));
   //inputMCfile = new TFile("out/tables_Glauber2012_AmptResponse_d20121115_v3.root","read");
   //inputMCfile = new TFile("out/tables_Glauber2012_HijingResponse_d20121115_v3.root","read");
   inputMCfile = new TFile("out/tables_Glauber2012B_EposLHCResponse_d20130118_v4.root","read");
   inputMCtable = (CentralityBins*)inputMCfile->Get(Form("CentralityTable_%s_SmearedGlauber_v4/run1",label.data()));
   //txtfile << "Using AMPT efficiency and AMPT smeared Glauber table" << endl << endl;
   txtfile << "Using EPOS efficiency and EPOS smeared Glauber table" << endl << endl;
   //txtfile << "Using HIJING efficiency and HIJING smeared Glauber table" << endl << endl;
 }

 //Setting up variables and branches
 double binboundaries[nbins+1];
 vector<float> values;
 TH1F * hist;
 if(!isMC) hist = new TH1F("hist","",hEff->GetNbinsX(),hEff->GetBinLowEdge(1),hEff->GetBinLowEdge(hEff->GetNbinsX()));

 float vtxZ, b, npart, ncoll, nhard, hf, hfplus, hfpluseta4, hfminuseta4, hfminus, hfhit, ee, eb;
 int run, npix, npixtrks, ntrks;
 t->SetBranchAddress("vz",&vtxZ);
 t->SetBranchAddress("run",&run);
 if(isMC){
    t->SetBranchAddress("b",&b);
    t->SetBranchAddress("Npart",	&npart);
    t->SetBranchAddress("Ncoll",	&ncoll);
    t->SetBranchAddress("Nhard",	&nhard);
 }
 t->SetBranchAddress("hiHF",		&hf);
 t->SetBranchAddress("hiHFplus",	&hfplus);
 t->SetBranchAddress("hiHFplusEta4",	&hfpluseta4);
 t->SetBranchAddress("hiHFminus",	&hfminus);
 t->SetBranchAddress("hiHFminusEta4",	&hfminuseta4);
 t->SetBranchAddress("hiHFhit",		&hfhit);
 t->SetBranchAddress("hiEE",		&ee);
 t->SetBranchAddress("hiEB",		&eb);
 t->SetBranchAddress("hiNpix",		&npix);
 t->SetBranchAddress("hiNpixelTracks",	&npixtrks);
 t->SetBranchAddress("hiNtracks",	&ntrks);
 //t->SetBranchAddress("hiNtracksOffline",	&ntrks);

 bool binB = label.compare("b") == 0;
 bool binNpart = label.compare("Npart") == 0;
 bool binNcoll = label.compare("Ncoll") == 0;
 bool binNhard = label.compare("Nhard") == 0;
 bool binHF = label.compare("HFtowers") == 0;
 bool binHFplus = label.compare("HFtowersPlus") == 0;
 bool binHFminus = label.compare("HFtowersMinus") == 0;
 bool binHFplusTrunc = label.compare("HFtowersPlusTrunc") == 0;
 bool binHFminusTrunc = label.compare("HFtowersMinusTrunc") == 0;
 bool binNpix = label.compare("PixelHits") == 0;
 bool binNpixTrks = label.compare("PixelTracks") == 0;
 bool binNtrks = label.compare("Tracks") == 0;

 //Event loop
 unsigned int Nevents = t->GetEntries();
 txtfile << "Number of events = " << Nevents << endl << endl;
 for(unsigned int iev = 0; iev < Nevents; iev++) {
   if(iev%10000 == 0) cout<<"Processing event: " << iev << endl;
   t->GetEntry(iev);

   //if(run!=runNum) continue;

   float parameter = -1;
   if(binB) parameter = b;
   if(binNpart) parameter = npart;
   if(binNcoll) parameter = ncoll;
   if(binNhard) parameter = nhard;
   if(binHF) parameter = hf;
   if(binHFplus) parameter = hfplus;
   if(binHFminus) parameter = hfminus;
   if(binHFplusTrunc) parameter = hfpluseta4;
   if(binHFminusTrunc) parameter = hfminuseta4;
   if(binNpix) parameter = npix;
   if(binNpixTrks) parameter = npixtrks;
   if(binNtrks) parameter = ntrks;

   values.push_back(parameter);

   if(!isMC) {
     hist->Fill(parameter);
   }

 }

 //Sorting the centrality variable vector
 if(binB) sort(values.begin(),values.end(),descend);
 else sort(values.begin(),values.end());

 //Finding the bin boundaries
 txtfile << "-------------------------------------" << endl;
 txtfile << label.data() << " based cuts are: " << endl;
 txtfile << "(";

 int size = values.size();
 binboundaries[nbins] = values[size-1];

 if(isMC) {
   for(int i = 0; i < nbins; i++) {
      int entry = (int)(i*(size/nbins));
      if(entry < 0 || i == 0) binboundaries[i] = 0;
      else binboundaries[i] = values[entry];
   }
 }
 else {
      TH1F * corr = (TH1F*)hist->Clone("corr");
      //TCanvas *c1 = new TCanvas();
      //c1->SetLogy();
      //corr->DrawCopy("hist");
      for (int j=1; j<corr->GetNbinsX(); j++) {
        if (hEff->GetBinContent(j) != 0) {
          corr->SetBinContent(j,corr->GetBinContent(j)/hEff->GetBinContent(j));
          corr->SetBinError(j,corr->GetBinError(j)/hEff->GetBinContent(j));
        }
      }
      //corr->SetLineColor(2);
      //corr->DrawCopy("hist same");
      //cout << "total integral = " << corr->Integral();
      float prev = 0;
      binboundaries[0] = 0;
      int j = 1;
      for (int i = 1; i < corr->GetNbinsX(); i++) {
        if(j>=nbins) continue;
        float a = corr->Integral(1,i,"");
        a = a/corr->Integral();
	//if(i<100) cout << i << " bin in x fraction of total integral = " << a << " j = " << j << endl;
        if (a > (float)j/nbins && prev < (float)j/nbins) {
		binboundaries[j] = corr->GetBinLowEdge(i+1);
		j++;
	}
        prev = a;
      }
 }
 for(int i = 0; i < nbins; i++) {
  if(binboundaries[i] < 0) binboundaries[i] = 0;
  txtfile << binboundaries[i] << ", ";
 }
 txtfile << binboundaries[nbins] << ")" << endl << "-------------------------------------" << endl;

 //***Determining Glauber results for MC and filling the table***
 if(isMC) {

  dir->cd();
  TH2D* hNpart = new TH2D("hNpart","",nbins,binboundaries,40,0,40);
  TH2D* hNcoll = new TH2D("hNcoll","",nbins,binboundaries,40,0,40);
  TH2D* hNhard = new TH2D("hNhard","",nbins,binboundaries,50,0,50);
  TH2D* hb = new TH2D("hb","",nbins,binboundaries,600,0,30);

  for(unsigned int iev = 0; iev < Nevents; iev++) {
     if( iev % 5000 == 0 ) cout<<"Processing event : " << iev << endl;
     t->GetEntry(iev);

     float parameter = -1;
     if(binB) parameter = b;
     if(binNpart) parameter = npart;
     if(binNcoll) parameter = ncoll;
     if(binNhard) parameter = nhard;
     if(binHF) parameter = hf;
     if(binHFplus) parameter = hfplus;
     if(binHFminus) parameter = hfminus;
     if(binHFplusTrunc) parameter = hfpluseta4;
     if(binHFminusTrunc) parameter = hfminuseta4;
     if(binNpix) parameter = npix;
     if(binNpixTrks) parameter = npixtrks;
     if(binNtrks) parameter = ntrks;

     hNpart->Fill(parameter,npart);
     hNcoll->Fill(parameter,ncoll);
     hNhard->Fill(parameter,nhard);
     hb->Fill(parameter,b);
     int bin = hNpart->GetXaxis()->FindBin(parameter) - 1;
     if(bin < 0) bin = 0;
     if(bin >= nbins) bin = nbins - 1;
     nt->Fill(parameter,bin,b,npart,ncoll,nhard);
  }

  TF1* fGaus = new TF1("fb","gaus(0)",0,2);
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

  txtfile<<"-------------------------------------"<<endl;
  txtfile<<"# Bin NpartMean NpartSigma NcollMean NcollSigma bMean bSigma BinEdge"<<endl;
  for(int i = 0; i < nbins; i++){
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

     txtfile << i << " " << hNpartMean->GetBinContent(ii) << " " << hNpartSigma->GetBinContent(ii) << " " << hNcollMean->GetBinContent(ii) << " " << hNcollSigma->GetBinContent(ii) << " " << hbMean->GetBinContent(ii) << " " <<hbSigma->GetBinContent(ii) << " " << binboundaries[ii-1] << " " <<endl;
  }
  txtfile<<"-------------------------------------"<<endl;

 } //***end of MC part***

 else { //***Data table with inputMCtable***

    txtfile<<"-------------------------------------"<<endl;
    txtfile<<"# Bin NpartMean NpartSigma NcollMean NcollSigma bMean bSigma BinEdge"<<endl;
    for(int i = 0; i < nbins; i++){
      int ii = nbins-i;
      bins->table_[i].n_part_mean = inputMCtable->NpartMeanOfBin(i);
      bins->table_[i].n_part_var = inputMCtable->NpartSigmaOfBin(i);
      bins->table_[i].n_coll_mean = inputMCtable->NcollMeanOfBin(i);
      bins->table_[i].n_coll_var = inputMCtable->NcollSigmaOfBin(i);
      bins->table_[i].b_mean = inputMCtable->bMeanOfBin(i);
      bins->table_[i].b_var = inputMCtable->bSigmaOfBin(i);
      bins->table_[i].n_hard_mean = inputMCtable->NhardMeanOfBin(i);
      bins->table_[i].n_hard_var = inputMCtable->NhardSigmaOfBin(i);
      bins->table_[i].ecc2_mean  = inputMCtable->eccentricityMeanOfBin(i);
      bins->table_[i].ecc2_var = inputMCtable->eccentricitySigmaOfBin(i);
      bins->table_[i].bin_edge = binboundaries[ii-1];

      txtfile << i << " " << bins->table_[i].n_part_mean << " " << bins->table_[i].n_part_var << " " << bins->table_[i].n_coll_mean << " " << bins->table_[i].n_coll_var << " " <<bins->table_[i].b_mean << " " << bins->table_[i].b_var << " " << bins->table_[i].n_hard_mean << " " << bins->table_[i].n_hard_var << " " << bins->table_[i].bin_edge << " " << endl;
    }
    txtfile<<"-------------------------------------"<<endl;

 } //***end of Data part***

 outFile->cd();
 dir->cd();
 bins->Write();
 nt->Write();  
 bins->Delete();
 outFile->Write();
 txtfile.close();

}

