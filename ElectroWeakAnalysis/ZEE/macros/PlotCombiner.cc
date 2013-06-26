/*
     Macro to make the plots .......................................

     Instructions:
     a. set up an input file that looks like the following:
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     # zee or wenu
     wenu
     # file name, type (sig, qcd, bce, gje, ewk), weight
     histos_wenu.root     sig     1.46
     histos_q20_30.root   qcd     0
     histos_q30_80.root   qcd     100.
     histos_q80_170.root  qcd     0
     histos_b20_30.root   bce     0
     histos_b30_80.root   bce     0
     histos_b80_170.root  bce     0
     histos_zee.root      ewk     0
     histos_wtaunu.root   ewk     0
     histos_ztautau.root  ewk     0
     histos_gj15.root     gje     0
     histos_gj20.root     gje     0
     histos_gj25.root     gje     10.12
     histos_gj30.root     gje     0
     histos_gj35.root     gje     0
     histos_wmunu.root    ewk     0
     histos_ttbar.root    ewk     0
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     lines that start with # are considered to be comments
     line 2 has wenu or zee. From line 4 the list of the histo files are listed
     (first word) then a type that could be sig,qcd,bce, gj or ewk in order to
     discriminate among different sources of bkgs and finally the weight that we
     want to weight the histogram entries. This particular example is for Wenu. For
     Zee one has to put type sig in the zee file and ewk in the Wenu file. The order
     of the files is arbitrary. Files with weight 0 will be ignored.
     After you have set up this code you run a root macro to combine the plots.
     You can do (not recommended - it actually crushes - to be debugged)
     root -b PlotCombiner.cc 
     or to compile it within root (recommended)
     root -b
     root [1] .L PlotCombiner.cc++
     root [2] PlotCombiner()
     
     and you finally get the plots.

     TO DO:
     functionalities to plot more kind of plots, e.g. efficiencies
     
     
     Further Questions/Contact:
     
         nikolaos.rompotis @ cern.ch



	 Nikolaos Rompotis - 29 June 09
	 18 Sept 09:  1st updgrade: input files in a text file
	 Imperial College London
	 
	 
*/


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "TString.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TLegend.h"

void plotMaker(TString histoName, TString typeOfplot,
	       vector<TString> file, vector<TString> type, 
	       vector<double> weight, 
	       TString xtitle, Int_t NBins, Double_t min, Double_t max);


void PlotCombiner()
{
  // read the file
  ifstream input("inputFiles");
  int i = 0;
  TString typeOfplot = "";
  vector<TString> types;
  vector<TString> files;
  vector<double> weights;

  if (input.is_open()) {
    std::string myline;
    while (! input.eof()) {
      getline(input, myline);
      TString line(myline);
      TString c('#');
      TString empty(' ');
      if (line[0] != c) {
	++i;
	if (i==1) typeOfplot=line;
	else {
	  // read until you find 3 words
	  TString fname("");
	  TString ftype("");
	  TString fw("");
	  int lineSize = (int) line.Length();
	  int j=0;
	  while (j<lineSize) {
	    if(line[j] != empty) fname += line[j];
	    else break;
	    ++j;
	  }
	  while (j<lineSize) {
	    if(line[j] != empty) ftype += line[j];
	    else if(ftype.Length()==3) break;
	    ++j;
	  }
	  while (j<lineSize) {
	    if(line[j] != empty) fw += line[j];
	    else{ if(fw.Length()>0) break;}
	    ++j;
	  }
	  if (fname.Length() == 0) break;
	  files.push_back(fname);
	  types.push_back(ftype);
	  double w = fw.Atof();
	  weights.push_back(w);
	  std::cout << fname << ", " << ftype << ", "<< w << std::endl;
	}
      }
    }
    input.close();
  }
  else {
    std::cout << "File with name inputFile was not found" << std::endl;
    return;
  }

  // now you can launch the jobs
  if (typeOfplot == "wenu") {
    cout << "wenu plot maker" << endl;
    //        ====================
    // =====> WHICH HISTOS TO PLOT
    //        ====================
    plotMaker("h_met", typeOfplot, files, types, weights, "MET (GeV)", 100,0,100);
  }
  else {
    cout << "zee plot maker" << endl;
    //        ====================
    // =====> WHICH HISTOS TO PLOT
    //        ====================
    plotMaker("h_mee", typeOfplot, files, types, weights, "M_{ee} (GeV)", 150,0,150);
  }


}



void plotMaker(TString histoName, TString wzsignal,
	       vector<TString> file, vector<TString> type, 
	       vector<double> weight, 
	       TString xtitle, Int_t NBins, Double_t min, Double_t max)
{
  gROOT->Reset();
  gROOT->ProcessLine(".L tdrstyle.C"); 
  gROOT->ProcessLine("setTDRStyle()");

  // Wenu Signal .......................................................
  TH1F h_wenu("h_wenu", "h_wenu", NBins, min, max);
  int fmax = (int) file.size();
  for (int i=0; i<fmax; ++i) {
    if (type[i] == "sig" && weight[i]>0) {
      TFile f(file[i]);
      TH1F *h = (TH1F*) f.Get(histoName);
      h_wenu.Add(h, weight[i]);
    }
  }
  // Bkgs ..............................................................
  //
  // QCD light flavor
  TH1F h_qcd("h_qcd", "h_qcd", NBins, min, max);
  for (int i=0; i<fmax; ++i) {
    if (type[i] == "qcd" && weight[i]>0) {
      TFile f(file[i]);
      TH1F *h = (TH1F*) f.Get(histoName);
      h_qcd.Add(h, weight[i]);
    }
  }
  // QCD heavy flavor
  TH1F h_bce("h_bce", "h_bce", NBins, min, max);
  for (int i=0; i<fmax; ++i) {
    if (type[i] == "bce" && weight[i]>0) {
      TFile f(file[i]);
      TH1F *h = (TH1F*) f.Get(histoName);
      h_bce.Add(h, weight[i]);
    }
  }
  // QCD Gjets
  TH1F h_gj("h_gj", "h_gj", NBins, min, max);
  for (int i=0; i<fmax; ++i) {
    if (type[i] == "gje" && weight[i]>0) {
      TFile f(file[i]);
      TH1F *h = (TH1F*) f.Get(histoName);
      h_gj.Add(h, weight[i]);
    }
  }
  // Other EWK bkgs
  TH1F h_ewk("h_ewk", "h_ewk", NBins, min, max);
  for (int i=0; i<fmax; ++i) {
    if (type[i] == "ewk" && weight[i]>0) {
      TFile f(file[i]);
      TH1F *h = (TH1F*) f.Get(histoName);
      h_ewk.Add(h, weight[i]);
    }
  }
  //
  // ok now decide how to plot them:
  // first the EWK bkgs
  h_ewk.SetFillColor(3);
  //
  // then the gjets
  h_gj.Add(&h_ewk);
  h_gj.SetFillColor(1);
  // thent the QCD dijets
  h_bce.Add(&h_qcd);
  h_bce.Add(&h_gj);
  h_bce.SetFillColor(2);
  // and the signal at last
  TH1F h_tot("h_tot", "h_tot", NBins, min, max);
  h_tot.Add(&h_bce);
  h_tot.Add(&h_wenu);
  h_wenu.SetLineColor(4);  h_wenu.SetLineWidth(2);
  //
  TCanvas c;
  h_tot.GetXaxis()->SetTitle(xtitle);
  h_tot.Draw("PE");
  h_bce.Draw("same");
  h_gj.Draw("same");
  h_ewk.Draw("same");
  h_wenu.Draw("same");

  // the Legend
  TLegend  leg(0.6,0.65,0.95,0.92);
  if (wzsignal == "wenu")
    leg.AddEntry(&h_wenu, "W#rightarrow e#nu","l");
  else
    leg.AddEntry(&h_wenu, "Z#rightarrow ee","l");
  leg.AddEntry(&h_tot, "Signal + Bkg","p");
  leg.AddEntry(&h_bce, "dijets","f");
  leg.AddEntry(&h_gj, "#gamma + jets","f");
  leg.AddEntry(&h_ewk,  "EWK+t#bar t", "f");
  leg.Draw("same");

  c.Print("test.png");



}



