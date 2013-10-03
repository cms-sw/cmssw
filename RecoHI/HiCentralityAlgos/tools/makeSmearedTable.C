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
#include <TFile.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TRandom.h>
#include <TTree.h>
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"

using namespace std;

bool descend(float i,float j) { return (i<j); }

const int nhist = 5;
TH2D * Npart_vs_PtGenPlusEta4[nhist];
TH2D * PtGenPlusEta4_vs_HFplusEta4[nhist];
TH2D * Npart_vs_PtGenMinusEta4[nhist];
TH2D * PtGenMinusEta4_vs_HFminusEta4[nhist];
TH2D * Npart_vs_Ngentrk[nhist];
TH2D * Ngentrk_vs_Ntracks[nhist];
TH1D * Proj1[1000];
TH1D * Proj2[1000];

float et = 0;

//------------------------------------------------------------------------
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
      hMean->SetBinError(i,func->GetParError(1)); //errors are not use later for the actual table
      hSigma->SetBinContent(i,h->GetRMS());
      hSigma->SetBinError(i,func->GetParError(2));

   }
}
//------------------------------------------------------------------------
void ProduceResponsePlots2(const char * infilename = "/tmp/azsigmon/HiForest_pPb_Hijing_NEWFIX_v2.root"){

 TH1::SetDefaultSumw2();

 TFile * infile = TFile::Open(infilename);
 TTree * tgen = (TTree *) infile->Get("HiGenParticleAna/hi");
 int mult;		tgen->SetBranchAddress("mult", &mult);
 float pt[10000];	tgen->SetBranchAddress("pt", pt);
 float eta[10000];	tgen->SetBranchAddress("eta", eta);
 float phi[10000];	tgen->SetBranchAddress("phi", phi);
 int pdg[10000];	tgen->SetBranchAddress("pdg", pdg);
 int chg[10000];	tgen->SetBranchAddress("chg", chg);
 int sta[10000];	tgen->SetBranchAddress("sta", sta);

 TTree * tskim = (TTree *) infile->Get("skimanalysis/HltTree");
 int selHFp;	tskim->SetBranchAddress("phfPosFilter1", &selHFp);
 int selHFm;	tskim->SetBranchAddress("phfNegFilter1", &selHFm);
 int selPix;	tskim->SetBranchAddress("phltPixelClusterShapeFilter", &selPix);
 int selVtx;	tskim->SetBranchAddress("pprimaryvertexFilter", &selVtx);

 TTree * thlt  = (TTree *) infile->Get("hltanalysis/HltTree");
 int trig;	thlt->SetBranchAddress("HLT_PAZeroBiasPixel_SingleTrack_v1", &trig);

 TTree * teven = (TTree *) infile->Get("hiEvtAnalyzer/HiTree");
 float b;		teven->SetBranchAddress("b", &b);
 float npart;		teven->SetBranchAddress("Npart", &npart);
 float ncoll;		teven->SetBranchAddress("Ncoll", &ncoll);
 float nhard;		teven->SetBranchAddress("Nhard", &nhard);
 int run;		teven->SetBranchAddress("run", &run);
 int lumi;		teven->SetBranchAddress("lumi", &lumi);
 float vtxZ;		teven->SetBranchAddress("vz",&vtxZ);
 float hf;		teven->SetBranchAddress("hiHF", &hf);
 float hfplus;		teven->SetBranchAddress("hiHFplus", &hfplus);
 float hfpluseta4;	teven->SetBranchAddress("hiHFplusEta4", &hfpluseta4);
 float hfminuseta4;	teven->SetBranchAddress("hiHFminusEta4", &hfminuseta4);
 float hfminus;		teven->SetBranchAddress("hiHFminus", &hfminus);
 float hfhit;		teven->SetBranchAddress("hiHFhit", &hfhit);
 float ee;		teven->SetBranchAddress("hiEE",	&ee);
 float eb;		teven->SetBranchAddress("hiEB",	 &eb);
 int npix;		teven->SetBranchAddress("hiNpix", &npix);
 int npixtrks;		teven->SetBranchAddress("hiNpixelTracks", &npixtrks);
 int ntrks;		teven->SetBranchAddress("hiNtracks", &ntrks);
 //int ntrks;		teven->SetBranchAddress("hiNtracksOffline", &ntrks);

 int NGenPlusEta4, NGenMinusEta4;
 double PtGenPlusEta4, PtGenMinusEta4;
 int NGenTrk;

 TFile * effFile = new TFile("out/efficiencies_Hijing_v3.root","read");
 TF1 * fitEff[nhist];
 for(int i=0; i<nhist; i++) {
  fitEff[i] = (TF1*)effFile->Get(Form("HFtowersPlusTrunc/fitEff%d",i));
 }

 //Create 1D and 2D histograms
 bool sel[nhist] = {0};

 for (int ih = 0; ih < nhist; ih++) {

  Npart_vs_PtGenPlusEta4[ih] = new TH2D(Form("Npart_vs_PtGenPlusEta4_%d",ih),";N_{part};#Sigma gen p_{T} in #eta > 4",40,0,40,500,0,500);
  PtGenPlusEta4_vs_HFplusEta4[ih] = new TH2D(Form("PtGenPlusEta4_vs_HFplusEta4_%d",ih),";#Sigma gen p_{T} in #eta > 4;#Sigma HF E_{T} in #eta > 4",500,0,500,200,0,200);
  Npart_vs_PtGenMinusEta4[ih] = new TH2D(Form("Npart_vs_PtGenMinusEta4_%d",ih),";N_{part};#Sigma gen p_{T} in #eta < -4",40,0,40,500,0,500);
  PtGenMinusEta4_vs_HFminusEta4[ih] = new TH2D(Form("PtGenMinusEta4_vs_HFminusEta4_%d",ih),";#Sigma gen p_{T} in #eta < -4;#Sigma HF E_{T} in #eta < -4",500,0,500,200,0,200);
  Npart_vs_Ngentrk[ih] = new TH2D(Form("Npart_vs_Ngentrk_%d",ih),";N_{part};# gen particles in |#eta|<2.4 p_{T}>0.4 GeV",40,0,40,500,0,500);
  Ngentrk_vs_Ntracks[ih] = new TH2D(Form("Ngentrk_vs_Ntracks_%d",ih),";# gen particles in |#eta|<2.4 p_{T}>0.4;hiNtracks",500,0,500,500,0,500);

 }

 //***event loop***
 int nEvents = teven->GetEntries();
 if (nEvents != tskim->GetEntries() || nEvents != thlt->GetEntries() || nEvents != tgen->GetEntries()) cout << "Error in number of events" << endl;

 for(int iev = 0; iev < nEvents; iev++) {

   for(int ih = 0; ih < nhist; ih++) sel[ih] = 0;
   NGenPlusEta4 = 0;
   PtGenPlusEta4 = 0;
   NGenMinusEta4 = 0;
   PtGenMinusEta4 = 0;
   NGenTrk = 0;

   if(iev%20000 == 0) cout << "Processing event: " << iev << endl;
   tskim->GetEntry(iev);
   thlt->GetEntry(iev);
   teven->GetEntry(iev);
   tgen->GetEntry(iev);

   //gen particle loop
   for(int i=0; i<mult; i++) {
     if (sta[i]==1 && eta[i] > 4.0 && eta[i] < 5.2) { //HFtowersPlusEta4
       NGenPlusEta4++;
       PtGenPlusEta4 += pt[i];
     }
     else if (sta[i]==1 && eta[i]<-4 && eta[i]>-5.2) {
       NGenMinusEta4++;
       PtGenMinusEta4 += pt[i];
     }
     else if (sta[i]==1 && eta[i]>-2.4 && eta[i]<2.4 && pt[i]>0.4) {
       NGenTrk++;
     }
   }

   for(int i=0; i<nhist; i++) {
     if (rand() > fitEff[i]->Eval(hfpluseta4)) sel[i] = 1;
   }

   for(int ih = 0; ih < nhist; ih++) {
    if(sel[ih]) {
      Npart_vs_PtGenPlusEta4[ih]->Fill(npart,PtGenPlusEta4);
      PtGenPlusEta4_vs_HFplusEta4[ih]->Fill(PtGenPlusEta4,hfpluseta4);
      Npart_vs_PtGenMinusEta4[ih]->Fill(npart,PtGenMinusEta4);
      PtGenMinusEta4_vs_HFminusEta4[ih]->Fill(PtGenMinusEta4,hfminuseta4);
      Npart_vs_Ngentrk[ih]->Fill(npart,NGenTrk);
      Ngentrk_vs_Ntracks[ih]->Fill(NGenTrk,ntrks);
    }
   }

 }//***end of event loop***
}
//------------------------------------------------------------------------
void getProjections(TH2D * h2, TH1D ** h1, const char* name, double merge, double mergePoint){
  
  if(!h2) return;
  for(int i = 0; i < h2->GetNbinsX()+1; i++){
    int imax = i;
    if(h2->GetBinCenter(i) > mergePoint) imax = i+merge;
    h1[i] = h2->ProjectionY(Form("%s_%d",name,i),i,imax);
    double integral = h1[i]->Integral();
    if(integral > 0){
      h1[i]->Scale(1./integral);
    }
  }

}
//------------------------------------------------------------------------
float getHFplusByPt(float Npart){
   
   int bin1 = Npart_vs_PtGenPlusEta4[0]->GetXaxis()->FindBin(Npart);
   if(bin1 > Npart_vs_PtGenPlusEta4[0]->GetNbinsX() ) bin1 = Npart_vs_PtGenPlusEta4[0]->GetNbinsX();
   if(bin1 < 1) bin1 = 1;
   et = Proj1[bin1]->GetRandom();

   int bin2 = PtGenPlusEta4_vs_HFplusEta4[0]->GetXaxis()->FindBin(et);
   if(bin2 > PtGenPlusEta4_vs_HFplusEta4[0]->GetNbinsX()) bin2 = PtGenPlusEta4_vs_HFplusEta4[0]->GetNbinsX();
   if(bin2 < 1) bin2 = 1;
   double hf = Proj2[bin2]->GetRandom();
   return hf;

}
//------------------------------------------------------------------------
float getHFminusByPt(float Npart){
   
   int bin1 = Npart_vs_PtGenMinusEta4[0]->GetXaxis()->FindBin(Npart);
   if(bin1 > Npart_vs_PtGenMinusEta4[0]->GetNbinsX() ) bin1 = Npart_vs_PtGenMinusEta4[0]->GetNbinsX();
   if(bin1 < 1) bin1 = 1;
   et = Proj1[bin1]->GetRandom();

   int bin2 = PtGenMinusEta4_vs_HFminusEta4[0]->GetXaxis()->FindBin(et);
   if(bin2 > PtGenMinusEta4_vs_HFminusEta4[0]->GetNbinsX()) bin2 = PtGenMinusEta4_vs_HFminusEta4[0]->GetNbinsX();
   if(bin2 < 1) bin2 = 1;
   double hf = Proj2[bin2]->GetRandom();
   return hf;

}
//------------------------------------------------------------------------
float getTracksByGen(float Npart){
   
   int bin1 = Npart_vs_Ngentrk[0]->GetXaxis()->FindBin(Npart);
   if(bin1 > Npart_vs_Ngentrk[0]->GetNbinsX() ) bin1 = Npart_vs_Ngentrk[0]->GetNbinsX();
   if(bin1 < 1) bin1 = 1;
   et = Proj1[bin1]->GetRandom();

   int bin2 = Ngentrk_vs_Ntracks[0]->GetXaxis()->FindBin(et);
   if(bin2 > Ngentrk_vs_Ntracks[0]->GetNbinsX()) bin2 = Ngentrk_vs_Ntracks[0]->GetNbinsX();
   if(bin2 < 1) bin2 = 1;
   double hf = Proj2[bin2]->GetRandom();
   return hf;

}
//------------------------------------------------------------------------
void makeSmearedTable(const int nbins = 100, const string label = "HFtowersPlusTrunc", const char * tag = "CentralityTable_HFtowersPlusTrunc_SmearedGlauber_sigma74_eff0_v5", int eff = 0) {

 TH1::SetDefaultSumw2();
 const char * inputMCforest = Form("/tmp/azsigmon/HiForest_pPb_Hijing_NEWFIX_v2.root");

 ProduceResponsePlots2(inputMCforest);

 bool plot = false;
 if(plot) {
   TCanvas *c1 = new TCanvas();
   Npart_vs_PtGenPlusEta4[0]->Draw("colz");
   TCanvas *c2 = new TCanvas();
   PtGenPlusEta4_vs_HFplusEta4[0]->Draw("colz");
   TCanvas *c3 = new TCanvas();
   Npart_vs_PtGenMinusEta4[0]->Draw("colz");
   TCanvas *c4 = new TCanvas();
   PtGenMinusEta4_vs_HFminusEta4[0]->Draw("colz");
   TCanvas *c5 = new TCanvas();
   Npart_vs_Ngentrk[0]->Draw("colz");
   TCanvas *c6 = new TCanvas();
   Ngentrk_vs_Ntracks[0]->Draw("colz");
 }

 bool binHFplusTrunc = label.compare("HFtowersPlusTrunc") == 0;
 bool binHFminusTrunc = label.compare("HFtowersMinusTrunc") == 0;
 bool binTracks = label.compare("Tracks") == 0;

 if (binHFplusTrunc) {
   getProjections(Npart_vs_PtGenPlusEta4[eff],Proj1,"Proj1",1,30);
   getProjections(PtGenPlusEta4_vs_HFplusEta4[eff],Proj2,"Proj2",1,140);
 }
 else if (binHFminusTrunc) {
   getProjections(Npart_vs_PtGenMinusEta4[0],Proj1,"Proj1",1,30);
   getProjections(PtGenMinusEta4_vs_HFminusEta4[0],Proj2,"Proj2",1,140);
 }
 else if (binTracks) {
   getProjections(Npart_vs_Ngentrk[0],Proj1,"Proj1",1,30);
   getProjections(Ngentrk_vs_Ntracks[0],Proj2,"Proj2",1,200);
 }

 //input Glauber ntuple
 const char * infilename = Form("/afs/cern.ch/work/t/tuos/public/pPb/Glauber/1M/Standard/Phob_Glau_pPb_sNN70mb_v15_1M_dmin04.root");
 //const char * infilename = Form("/afs/cern.ch/work/t/tuos/public/pPb/Glauber/1M/D/D04914/Phob_Glau_pPb_sNN70mb_v15_1M_D04914.root");
 //const char * infilename = Form("/afs/cern.ch/work/t/tuos/public/pPb/Glauber/1M/D/D06006/Phob_Glau_pPb_sNN70mb_v15_1M_D06006.root");
 //const char * infilename = Form("/afs/cern.ch/work/t/tuos/public/pPb/Glauber/1M/R/R649/Phob_Glau_pPb_sNN70mb_v15_1M_R649.root");
 //const char * infilename = Form("/afs/cern.ch/work/t/tuos/public/pPb/Glauber/1M/R/R675/Phob_Glau_pPb_sNN70mb_v15_1M_R675.root");
 //const char * infilename = Form("/afs/cern.ch/work/t/tuos/public/pPb/Glauber/1M/dmin/dmin00/Phob_Glau_pPb_sNN70mb_v15_1M_dmin00.root");
 //const char * infilename = Form("/afs/cern.ch/work/t/tuos/public/pPb/Glauber/1M/dmin/dmin08/Phob_Glau_pPb_sNN70mb_v15_1M_dmin08.root");
 //const char * infilename = Form("/afs/cern.ch/work/t/tuos/public/pPb/Glauber/1M/sigma/sigma66/Phob_Glau_pPb_sNN66mb_v15_1M_dmin04.root");
 //const char * infilename = Form("/afs/cern.ch/work/t/tuos/public/pPb/Glauber/1M/sigma/sigma74/Phob_Glau_pPb_sNN74mb_v15_1M_dmin04.root");
 TChain * t = new TChain("nt_p_Pb");
 t->Add(infilename);

 //output
 //const char* outfilename = "out/tables_Glauber2012B_AmptResponse_d20130116_v4.root";
 //const char* outfilename = "out/tables_Glauber2012B_EposLHCResponse_d20130118_v4.root";
 const char* outfilename = "out/tables_Glauber2012B_HijingResponse_d20130130_v5.root";
 TFile * outf = new TFile(outfilename,"update");
 outf->cd();
 TDirectory* dir = outf->mkdir(tag);
 dir->cd();
 TNtuple* nt = new TNtuple("nt","","HFbyPt:genPt:Bin:b:Npart:Ncoll:Nhard");
 CentralityBins * outputTable = new CentralityBins(Form("run%d",1), tag, nbins);
 outputTable->table_.reserve(nbins);

 ofstream txtfile("out/output.txt");
 txtfile << "Input Glauber tree: " << infilename << endl << "Input MC HiForest HIJING" << endl;

 //Setting up variables and branches
 double binboundaries[nbins+1];
 vector<float> values;

 float b, npart, ncoll, nhard, parameter;

 t->SetBranchAddress("B",&b);
 t->SetBranchAddress("Npart",&npart);
 t->SetBranchAddress("Ncoll",&ncoll);
 nhard = 0;

 //Event loop 1
 unsigned int Nevents = t->GetEntries();
 txtfile << "Number of events = " << Nevents << endl << endl;
 for(unsigned int iev = 0; iev < Nevents; iev++) {
   if(iev%20000 == 0) cout<<"Processing event: " << iev << endl;
   t->GetEntry(iev);

   if (binHFplusTrunc) parameter = getHFplusByPt(npart);
   if (binHFminusTrunc) parameter = getHFminusByPt(npart);
   if (binTracks) parameter = getTracksByGen(npart);

   values.push_back(parameter);
 }

 if(label.compare("b") == 0) sort(values.begin(),values.end(),descend);
 else sort(values.begin(),values.end());

 //Finding the bin boundaries
 txtfile << "-------------------------------------" << endl;
 txtfile << label.data() << " based cuts are: " << endl;
 txtfile << "(";

 int size = values.size();
 binboundaries[nbins] = values[size-1];

 for(int i = 0; i < nbins; i++) {
   int entry = (int)(i*(size/nbins));
   if(entry < 0 || i == 0) binboundaries[i] = 0;
   else binboundaries[i] = values[entry];
   txtfile << binboundaries[i] << ", ";
 }
 txtfile << binboundaries[nbins] << ")" << endl << "-------------------------------------" << endl;

 // Determining Glauber results in various bins
 TH2D* hNpart = new TH2D("hNpart","",nbins,binboundaries,40,0,40);
 TH2D* hNcoll = new TH2D("hNcoll","",nbins,binboundaries,40,0,40);
 TH2D* hNhard = new TH2D("hNhard","",nbins,binboundaries,50,0,50);
 TH2D* hb = new TH2D("hb","",nbins,binboundaries,600,0,30);

 for(unsigned int iev = 0; iev < Nevents; iev++) {
   if( iev % 20000 == 0 ) cout<<"Processing event : " << iev << endl;
   t->GetEntry(iev);

   if (binHFplusTrunc) parameter = getHFplusByPt(npart);
   if (binHFminusTrunc) parameter = getHFminusByPt(npart);
   if (binTracks) parameter = getTracksByGen(npart);

   hNpart->Fill(parameter,npart);
   hNcoll->Fill(parameter,ncoll);
   hNhard->Fill(parameter,nhard);
   hb->Fill(parameter,b);

   int bin = hNpart->GetXaxis()->FindBin(parameter) - 1;
   if(bin < 0) bin = 0;
   if(bin >= nbins) bin = nbins - 1;
   nt->Fill(parameter, et, bin, b, npart, ncoll, nhard);
 }

 TCanvas *cf = new TCanvas();
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
   outputTable->table_[i].n_part_mean = hNpartMean->GetBinContent(ii);
   outputTable->table_[i].n_part_var = hNpartSigma->GetBinContent(ii);
   outputTable->table_[i].n_coll_mean = hNcollMean->GetBinContent(ii);
   outputTable->table_[i].n_coll_var = hNcollSigma->GetBinContent(ii);
   outputTable->table_[i].b_mean = hbMean->GetBinContent(ii);
   outputTable->table_[i].b_var = hbSigma->GetBinContent(ii);
   outputTable->table_[i].n_hard_mean = hNhardMean->GetBinContent(ii);
   outputTable->table_[i].n_hard_var = hNhardSigma->GetBinContent(ii);
   outputTable->table_[i].bin_edge = binboundaries[ii-1];

   txtfile << i << " " << hNpartMean->GetBinContent(ii) << " " << hNpartSigma->GetBinContent(ii) << " " << hNcollMean->GetBinContent(ii) << " " << hNcollSigma->GetBinContent(ii) << " " << hbMean->GetBinContent(ii) << " " <<hbSigma->GetBinContent(ii) << " " << binboundaries[ii-1] << " " <<endl;
 }
 txtfile<<"-------------------------------------"<<endl;

 outf->cd();
 dir->cd();
 outputTable->Write();
 nt->Write();
 for(int ih = 0; ih < nhist; ih++) {
   Npart_vs_PtGenPlusEta4[ih]->Write();
   PtGenPlusEta4_vs_HFplusEta4[ih]->Write();
   Npart_vs_PtGenMinusEta4[ih]->Write();
   PtGenMinusEta4_vs_HFminusEta4[ih]->Write();
   Npart_vs_Ngentrk[ih]->Write();
   Ngentrk_vs_Ntracks[ih]->Write();
 }
 outf->Write();
 txtfile.close();

}
