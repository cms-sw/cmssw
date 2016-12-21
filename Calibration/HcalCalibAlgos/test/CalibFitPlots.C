//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L CalibFitPlots.C+g
//             For standard set of histograms from CalibMonitor
//  FitHistStandard(infile, outfile, prefix, mode, append, saveAll);
//      Defaults: mode=11111, append=true, saveAll=false
//
//             For extended set of histograms from CalibMonitor
//  FitHistExtended(infile, outfile, prefix, numb, append);
//      Defaults: numb=50, append=true
//
//             For plotting stored histograms from FitHist's
//  PlotHist(infile, prefix, text, modePlot, save);
//      Defaults: modePlot=0, save=false
//
//             For plotting stored histograms from CalibTree
//  PlotHistCorrResults(infile, text, save);
//      Defaults: save=false
//
//             For plotting correction factors
//  PlotHistCorrFactor(infile, text, nmin, save);
//      Defaults: nmin=20, save=false
//
//             For plotting correction factors from 2 different runs on the
//             same canvas
//  PlotHistCorrFactors(infile1, text1, infile2, text2, ratio, drawStatBox,
//                      nmin, save)
//      Defaults: drawStatBox=true, nmin=100, save=false
//
//             For plotting correction factors including systematics
//  PlotHistCorrSys(infilec, conds, text, 
//      Defaults: save=false
//
//  where:
//  infile   (std::string)  = Name of the input ROOT file
//  outfile  (std::string)  = Name of the output ROOT file
//  prefix   (std::string)  = Prefix for the histogram names
//  mode     (int)          = Flag to check which set of histograms to be
//                            done. It has the format lthdo where each of 
//                            l, t,h,d,o can have a value 0 or 1 to select
//                            or deselect. l,t,h,d,o for momentum range
//                            60-100, 30-40, all, 20-30, 40-60 Gev (11111)
//  append   (bool)         = Open the output in Update/Recreate mode (True)
//  saveAll  (bool)         = Flag to save intermediate plots (False)
//  numb     (int)          = Number of eta bins (42 for -21:21)
//  text     (std::string)  = Extra text to be put in the text title
//  modePlot (int)          = Flag to plot E/p distribution (0) or <E/p> as
//                            a function of ieta (1)
//  save     (bool)         = if true it saves the canvas as a pdf file
//  nmin     (int)          = minimum # of #ieta points needed to show the
//                            fitted line
//  ratio    (bool)         = set to show the ratio plot (false)
//  drawStatBox (bool)      = set to show the statistical box (true)
//  infilc   (string)       = prefix of the file names of correction factors
//                            (assumes file name would be the prefix followed
//                            by _condX.txt where X=0 for the default version
//                            and 1..conds for the variations)
//  conds    (int)          = number of variations in estimating systematic
//                            checks
//////////////////////////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TProfile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdlib>

struct cfactors {
  int    ieta, depth;
  double corrf, dcorr;
  cfactors(int ie=0, int dp=0, double cor=1, double dc=0) :
    ieta(ie), depth(dp), corrf(cor), dcorr(dc) {};
};

std::pair<double,double> GetMean(TH1D* hist, double xmin, double xmax) {
  
  double mean(0), rms(0), err(0), wt(0);
  for (int i=1; i<=hist->GetNbinsX(); ++i) {
    if (((hist->GetBinLowEdge(i)) >= xmin) || 
	((hist->GetBinLowEdge(i)+hist->GetBinWidth(i)) <= xmax)) {
      double cont = hist->GetBinContent(i);
      double valu = hist->GetBinLowEdge(i)+0.5*+hist->GetBinWidth(i);
      wt         += cont;
      mean       += (valu*cont);
      rms        += (valu*valu*cont);
    }
  }
  if (wt > 0) {
    mean /= wt;
    rms  /= wt;
    err   = std::sqrt((rms-mean*mean)/wt);
  }
  return std::pair<double,double>(mean,err);
}

std::vector <std::string> splitString (const std::string& fLine) {
  std::vector <std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine [i] == ' ' || i == fLine.size ()) {
      if (!empty) {
	std::string item (fLine, start, i-start);
	result.push_back (item);
	empty = true;
      }
      start = i+1;
    } else {
      if (empty) empty = false;
    }
  }
  return result;
}

void FitHistStandard(std::string infile,std::string outfile,std::string prefix,
		     int mode=11111, bool append=true, bool saveAll=false) {

  int iname[5]      = {0, 1, 2, 3, 4};
  int checkmode[5]  = {10, 1000, 1, 10000, 100};
  double xbins[9]   = {-21.0, -16.0, -12.0, -6.0, 0.0, 6.0, 12.0, 16.0, 21.0};
  double vbins[6]   = {0.0, 7.0, 10.0, 13.0, 16.0, 50.0};
  double dlbins[9]  = {0.0, 0.10, 0.20, 0.50, 1.0, 2.0, 2.5, 3.0, 10.0};
  std::string sname[4] = {"ratio","etaR", "dl1R","nvxR"};
  std::string lname[4] = {"Z", "E", "L", "V"};
  std::string xname[4] = {"i#eta", "i#eta", "d_{L1}", "# Vertex"};
  int         numb[4]  = {8, 8, 8, 5};
  bool        debug(false);

  TFile      *file = new TFile(infile.c_str());
  std::vector<TH1D*> hists;
  char name[100];
  if (file != 0) {
    for (int m1=0; m1<4; ++m1) {
      for (int m2=0; m2<5; ++m2) {
	sprintf (name, "%s%s%d0", prefix.c_str(), sname[m1].c_str(), iname[m2]);
	TH1D* hist0 = (TH1D*)file->FindObjectAny(name);
	bool ok = ((hist0 != 0) && (hist0->GetEntries() > 25));
	if ((mode/checkmode[m2])%10 > 0 && ok) {
	  TH1D* histo(0);
	  sprintf (name, "%s%s%d", prefix.c_str(), lname[m1].c_str(),iname[m2]);
	  if (m1 <= 1)      histo = new TH1D(name, hist0->GetTitle(), numb[m1], xbins);
	  else if (m1 == 2) histo = new TH1D(name, hist0->GetTitle(), numb[m1], dlbins);
	  else              histo = new TH1D(name, hist0->GetTitle(), numb[m1], vbins);
	  for (int j=0; j<=numb[m1]; ++j) {
	    sprintf (name, "%s%s%d%d", prefix.c_str(), sname[m1].c_str(), iname[m2], j);
	    TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
	    TH1D* hist  = (TH1D*)hist1->Clone();
	    double value(0), error(0);
	    if (hist->GetEntries() > 0) {
	      value = hist->GetMean(); error = hist->GetRMS();
	    }
	    if (hist->GetEntries() > 4) {
	      double mean = hist->GetMean(), rms = hist->GetRMS();
	      double LowEdge = mean - 1.5*rms;
	      double HighEdge = mean + 2.0*rms;
	      if (LowEdge < 0.15) LowEdge = 0.15;
	      char option[20];
	      if (hist0->GetEntries() > 100) sprintf (option, "+QRS");
	      else                           sprintf (option, "+QRWLS");
	      double minvalue(0.30);
	      TFitResultPtr Fit = hist->Fit("gaus",option,"",LowEdge,HighEdge);
	      value = Fit->Value(1);
	      error = Fit->FitResult::Error(1); 
	      std::pair<double,double> meaner = GetMean(hist,0.2,2.0);
	      if (debug) std::cout << "Fit " << value << ":" << error << ":" 
				   << hist->GetMeanError() << " Mean " 
				   << meaner.first << ":" << meaner.second;
	      if (value < minvalue || value > 2.0 || error > 0.5) {
		value = meaner.first; error = meaner.second;
	      }
	      if (debug) std::cout << " Final " << value << ":" << error 
				   << std::endl;
	    }
	    if (j == 0) {
	      hists.push_back(hist);
	    } else {
	      if (saveAll) hists.push_back(hist);
	      histo->SetBinContent(j, value);
	      histo->SetBinError(j, error);
	    }
	  }
	  if (histo->GetEntries() > 2) {
	    int    nbin    = histo->GetNbinsX();
	    double LowEdge = histo->GetBinLowEdge(1);
	    double HighEdge= histo->GetBinLowEdge(nbin)+histo->GetBinWidth(nbin);
	    TFitResultPtr Fit = histo->Fit("pol0","+QRWLS","",LowEdge,HighEdge);
	    if (debug) std::cout << "Fit to Pol0: " << Fit->Value(0) << " +- "
				 << Fit->FitResult::Error(0) << std::endl;
	    histo->GetXaxis()->SetTitle(xname[m1].c_str());
	    histo->GetYaxis()->SetTitle("<E_{HCAL}/(p-E_{ECAL})>");
	    histo->GetYaxis()->SetRangeUser(0.4,1.6);
	  }
	  hists.push_back(histo);
	}
      }
    }
    TFile* theFile(0);
    if (append) {
      theFile = new TFile(outfile.c_str(), "UPDATE");
    } else {
      theFile = new TFile(outfile.c_str(), "RECREATE");
    }
    theFile->cd();
    for (unsigned int i=0; i<hists.size(); ++i) {
      TH1D* hnew = (TH1D*)hists[i]->Clone();
      hnew->Write();
    }
    theFile->Close();
    file->Close();
  }
}

void FitHistExtended(std::string infile, std::string outfile,std::string prefix,
		     int numb=50, bool append=true) {
  std::string sname("ratio"), lname("Z");
  int         iname(2);
  bool        debug(false);
  double      xbins[99];
  int         neta = numb/2;
  for (int k=0; k<neta; ++k) {
    xbins[k]      = (k-neta)-0.5;
    xbins[numb-k] = (neta-k) + 0.5;
  }
  xbins[neta] = 0;
  TFile      *file = new TFile(infile.c_str());
  std::vector<TH1D*> hists;
  char name[200];
  if (debug) std::cout << infile << " " << file << std::endl;
  if (file != 0) {
    sprintf (name, "%s%s%d0", prefix.c_str(), sname.c_str(), iname);
    TH1D* hist0 = (TH1D*)file->FindObjectAny(name);
    bool  ok   = (hist0 != 0);
    if (debug) std::cout << name << " Pointer " << hist0 << " " << ok 
			 << std::endl;
    if (ok) {
      sprintf (name, "%s%s%d", prefix.c_str(), lname.c_str(), iname);
      TH1D* histo = new TH1D(name, hist0->GetTitle(), numb, xbins);

      int   nbin = hist0->GetNbinsX();
      if (hist0->GetEntries() > 10) {
	double mean = hist0->GetMean(), rms = hist0->GetRMS();
	double LowEdge = mean - 1.5*rms;
	double HighEdge = mean + 2.0*rms;
	if (LowEdge < 0.15) LowEdge = 0.15;
	char option[20];
	if (hist0->GetEntries() > 100) {
	  sprintf (option, "+QRS");
	} else {
	  sprintf (option, "+QRWLS");
	  HighEdge= mean+1.5*rms;
	}
	TFitResultPtr Fit = hist0->Fit("gaus",option,"",LowEdge,HighEdge);
	std::pair<double,double> meaner = GetMean(hist0,0.2,2.0);
	if (debug) std::cout << "Fit " << Fit->Value(1) << ":" 
			     << Fit->FitResult::Error(1) << ":" 
			     << hist0->GetMeanError() << " Mean " 
			     << meaner.first << ":" << meaner.second;
      }
      int         nv1(100), nv2(0);
      for (int j=0; j<=numb; ++j) {
	sprintf (name, "%s%s%d%d", prefix.c_str(), sname.c_str(), iname, j);
	TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
	TH1D* hist  = (TH1D*)hist1->Clone();
	double value(0), error(0), total(0);
	if (hist->GetEntries() > 0) {
	  value = hist->GetMean(); error = hist->GetRMS();
	  for (int i=1; i<=nbin; ++i) total += hist->GetBinContent(i);
	}
	if (total > 4) {
	  if (nv1 > j) nv1 = j;
	  if (nv2 < j) nv2 = j;
	  double mean = hist->GetMean(), rms = hist->GetRMS();
	  double LowEdge = mean - 1.5*rms;
	  double HighEdge = mean + 2.0*rms;
	  if (LowEdge < 0.15) LowEdge = 0.15;
	  char option[20];
	  if (total > 100) {
	    sprintf (option, "+QRS");
	  } else {
            sprintf (option, "+QRWLS");
	    HighEdge= mean+1.5*rms;
	  }
	  double minvalue(0.30);
	  TFitResultPtr Fit = hist->Fit("gaus",option,"",LowEdge,HighEdge);
	  value = Fit->Value(1);
	  error = Fit->FitResult::Error(1); 
	  std::pair<double,double> meaner = GetMean(hist,0.2,2.0);
	  if (debug) std::cout << "Fit " << value << ":" << error << ":" 
			       << hist->GetMeanError() << " Mean " 
			       << meaner.first << ":" << meaner.second;
	  if (value < minvalue || value > 2.0 || error > 0.5) {
	    value = meaner.first; error = meaner.second;
	  }
	  if (debug) std::cout << " Final " << value << ":" << error<<std::endl;
	}
	hists.push_back(hist);
	if (j != 0) {
	  histo->SetBinContent(j, value);
	  histo->SetBinError(j, error);
	}
      }
      if (histo->GetEntries() > 2) {
	TFitResultPtr Fit = histo->Fit("pol0","+QRWLS","",xbins[nv1],xbins[nv2]);
	if (debug) std::cout << "Fit to Pol0: " << Fit->Value(0) << " +- "
			     << Fit->FitResult::Error(0) << " in range " << nv1
			     << ":" << xbins[nv1] << ":" << nv2 << ":" 
			     << xbins[nv2] << std::endl;
	histo->GetXaxis()->SetTitle("i#eta");
	histo->GetYaxis()->SetTitle("<E_{HCAL}/(p-E_{ECAL})>");
	histo->GetYaxis()->SetRangeUser(0.4,1.6);
      }
      hists.push_back(histo);
    }
    TFile* theFile(0);
    if (append) {
      theFile = new TFile(outfile.c_str(), "UPDATE");
    } else {
      theFile = new TFile(outfile.c_str(), "RECREATE");
    }

    theFile->cd();
    for (unsigned int i=0; i<hists.size(); ++i) {
      TH1D* hnew = (TH1D*)hists[i]->Clone();
      hnew->Write();
    }
    theFile->Close();
    file->Close();
  }
}

void PlotHist(std::string infile, std::string prefix, std::string text,
	      int mode=0, bool save=false) {

  std::string name0[5] = {"ratio00","ratio10","ratio20","ratio30","ratio40"};
  std::string name1[5] = {"Z0", "Z1", "Z2", "Z3", "Z4"};
  std::string name2[5] = {"L0", "L1", "L2", "L3", "L4"};
  std::string name3[5] = {"V0", "V1", "V2", "V3", "V4"};
  std::string title[5] = {"Tracks with p = 20:30 GeV",
			  "Tracks with p = 30:40 GeV",
			  "Tracks with p = 40:60 GeV",
			  "Tracks with p = 60:100 GeV",
			  "Tracks with p = 20:100 GeV"};
  std::string xtitl[4] = {"E_{HCAL}/(p-E_{ECAL})","i#eta","d_{L1}","# Vertex"};
  std::string ytitl[4] = {"Tracks","<E_{HCAL}/(p-E_{ECAL})>",
			  "<E_{HCAL}/(p-E_{ECAL})>","<E_{HCAL}/(p-E_{ECAL})>"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  int iopt(1110);
  if (mode < 0 || mode > 3) mode = 0;
  if (mode != 0) iopt = 10;
  gStyle->SetOptStat(iopt);  gStyle->SetOptFit(1);
  TFile      *file = new TFile(infile.c_str());
  char name[100], namep[100];
  for (int k=0; k<5; ++k) {
    if (mode == 1) {
      sprintf (name, "%s%s", prefix.c_str(), name1[k].c_str());
    } else if (mode == 2) {
      sprintf (name, "%s%s", prefix.c_str(), name2[k].c_str());
    } else if (mode == 3) {
      sprintf (name, "%s%s", prefix.c_str(), name3[k].c_str());
    } else {
      sprintf (name, "%s%s", prefix.c_str(), name0[k].c_str());
    }
    TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
    if (hist1 != 0) {
      TH1D* hist = (TH1D*)(hist1->Clone()); 
      sprintf (namep, "c_%s", name);
      TCanvas *pad = new TCanvas(namep, namep, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      hist->GetXaxis()->SetTitle(xtitl[mode].c_str());
      hist->GetYaxis()->SetTitle(ytitl[mode].c_str());
      hist->GetYaxis()->SetLabelOffset(0.005);
      hist->GetYaxis()->SetTitleOffset(1.20);
      if (mode == 0) hist->GetXaxis()->SetRangeUser(0.0,2.5);
      hist->Draw();
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	double ymin = (mode == 0) ? 0.70 : 0.80; 
	st1->SetY1NDC(ymin); st1->SetY2NDC(0.90);
	st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
      }
      TPaveText *txt1 = new TPaveText(0.50,0.60,0.90,0.65,"blNDC");
      txt1->SetFillColor(0);
      char txt[100];
      if (text == "") sprintf (txt, "%s", title[k].c_str());
      else            sprintf (txt, "%s (%s)", title[k].c_str(), text.c_str());
      txt1->AddText(txt);
      txt1->Draw("same");
      pad->Modified();
      pad->Update();
      if (save) {
	sprintf (name, "%s.pdf", pad->GetName());
	pad->Print(name);
      }	
    }
  }
}

void PlotHistCorrResults(std::string infile, std::string text, bool save=false){

  std::string name[5]  = {"Eta1Bf","Eta2Bf","Eta1Af","Eta2Af","Cvg0"};
  std::string title[5] = {"Mean at the start of itertions",
			  "Median at the start of itertions",
			  "Mean at the end of itertions",
			  "Median at the end of itertions",""};
  int         type[5]  = {0,0,0,0,1};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(10);         gStyle->SetOptFit(10);
  TFile      *file = new TFile(infile.c_str());
  char namep[100];
  for (int k=0; k<5; ++k) {
    TH1D* hist1 = (TH1D*)file->FindObjectAny(name[k].c_str());
    if (hist1 != 0) {
      TH1D* hist = (TH1D*)(hist1->Clone()); 
      sprintf (namep, "c_%s", name[k].c_str());
      TCanvas *pad = new TCanvas(namep, namep, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      hist->GetYaxis()->SetLabelOffset(0.005);
      hist->GetYaxis()->SetTitleOffset(1.20);
      double xmin = hist->GetBinLowEdge(1);
      int    nbin = hist->GetNbinsX();
      double xmax = hist->GetBinLowEdge(nbin)+hist->GetBinWidth(nbin);
      std::cout << hist->GetTitle() << " Bins " << nbin << ":" << xmin << ":" << xmax << std::endl;
      double xlow(0.12), ylow(0.82);
      char txt[100], option[2];
      if (type[k] == 0) {
	sprintf (namep, "f_%s", name[k].c_str());
	TF1* func   = new TF1(namep, "pol0", xmin, xmax);
	hist->Fit(func,"+QWL","");
	if (text == "") sprintf (txt, "%s", title[k].c_str());
	else            sprintf (txt, "%s (balancing the %s)", title[k].c_str(), text.c_str());
	sprintf (option, "%s", "");
      } else {
	hist->GetXaxis()->SetTitle("Iterations");
	hist->GetYaxis()->SetTitle("Convergence");
	hist->SetMarkerStyle(20);
	hist->SetMarkerColor(2);
	hist->SetMarkerSize(0.8);
	xlow = 0.50;
	ylow = 0.86;
	sprintf (txt, "(%s)", text.c_str());
	sprintf (option, "%s", "p");
      }
      hist->Draw(option);
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetY1NDC(ylow); st1->SetY2NDC(0.90);
	st1->SetX1NDC(0.70); st1->SetX2NDC(0.90);
      }
      TPaveText *txt1 = new TPaveText(xlow,0.82,0.68,0.88,"blNDC");
      txt1->SetFillColor(0);
      txt1->AddText(txt);
      txt1->Draw("same");
      pad->Modified();
      pad->Update();
      if (save) {
	sprintf (namep, "%s.pdf", pad->GetName());
	pad->Print(namep);
      }	
    }
  }
}

void PlotHistCorrFactor(std::string infile, std::string text, int nmin=20,
			bool save=false) {

  std::vector<cfactors> cfacs;
  std::ifstream fInput(infile.c_str());
  int etamin(100), etamax(-100), maxdepth(0);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer [1024];
    unsigned int all(0), good(0);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      if (buffer [0] == '#') continue; //ignore comment
      std::vector <std::string> items = splitString (std::string (buffer));
      if (items.size () != 5) {
	std::cout << "Ignore  line: " << buffer << std::endl;
      } else {
	++good;
	int   ieta  = std::atoi (items[1].c_str());
	int   depth = std::atoi (items[2].c_str());
	float corrf = std::atof (items[3].c_str());
	float dcorr = std::atof (items[4].c_str());
	cfactors cfac(ieta,depth,corrf,dcorr);
	cfacs.push_back(cfac);
	if (ieta > etamax) etamax = ieta;
	if (ieta < etamin) etamin = ieta;
	if (depth > maxdepth) maxdepth = depth;
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records"
	      << std::endl;
  }
  /*
  for (unsigned int k = 0; k < cfacs.size(); ++k) 
    std::cout << "[" << k << "] " << cfacs[k].ieta << " " 
	      << cfacs[k].depth << " " << cfacs[k].corrf
	      << " " << cfacs[k].dcorr << std::endl;
  */

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(10);         gStyle->SetOptFit(10);
  int colors[6] = {1,6,4,7,2,9};
  int mtype[6]  = {20,21,22,23,24,33};
  int nbin = etamax - etamin + 1;
  std::vector<TH1D*> hists;
  std::vector<int>   entries;
  char               name[100];
  double             dy(0);
  int                fits(0);
  for (int j=0; j<maxdepth; ++j) {
    sprintf (name, "hd%d", j+1);
    TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
    int nent(0);
    for (unsigned int k = 0; k < cfacs.size(); ++k) {
      if (cfacs[k].depth == j+1) {
	int ieta = cfacs[k].ieta;
	int bin  = ieta - etamin + 1;
	float val = cfacs[k].corrf;
	float dvl = cfacs[k].dcorr;
	h->SetBinContent(bin,val);
	h->SetBinError(bin,dvl);
	nent++;
      }
    }
    if (nent > nmin) {
      fits++;
      dy  += 0.025;
      sprintf (name, "hdf%d", j+1);
      TF1* func   = new TF1(name, "pol0", etamin, etamax);
      h->Fit(func,"+QWLR","");
    }
    h->SetLineColor(colors[j]);
    h->SetMarkerColor(colors[j]);
    h->SetMarkerStyle(mtype[j]);
    h->GetXaxis()->SetTitle("i#eta");
    h->GetYaxis()->SetTitle("Correction Factor");
    h->GetYaxis()->SetLabelOffset(0.005);
    h->GetYaxis()->SetTitleOffset(1.20);
    h->GetYaxis()->SetRangeUser(0.0,2.0);
    hists.push_back(h);
    entries.push_back(nent);
    dy  += 0.025;
  }
  TCanvas *pad = new TCanvas("CorrFactor","CorrFactor", 700, 500);
  pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
  double yh = 0.90;
  double yl = yh-0.025*hists.size()-dy-0.01;
  TLegend *legend = new TLegend(0.60, yl, 0.90, yl+0.025*hists.size());
  legend->SetFillColor(kWhite);
  for (unsigned int k=0; k<hists.size(); ++k) {
    if (k == 0) hists[k]->Draw("");
    else        hists[k]->Draw("sames");
    pad->Update();
    TPaveStats* st1 = (TPaveStats*)hists[k]->GetListOfFunctions()->FindObject("stats");
    if (st1 != NULL) {
      dy = (entries[k] > nmin) ? 0.05 : 0.025;
      st1->SetLineColor(colors[k]);
      st1->SetTextColor(colors[k]);
      st1->SetY1NDC(yh-dy); st1->SetY2NDC(yh);
      st1->SetX1NDC(0.70); st1->SetX2NDC(0.90);
      yh -= dy;
    }
    sprintf (name, "Depth %d (%s)", k+1, text.c_str());
    legend->AddEntry(hists[k],name,"lp");
  }
  legend->Draw("same");
  pad->Update();
  if (fits < 1) {
    pad->Range(0.0,0.0,1.0,1.0);
    TLine line = TLine(0.0,0.5,1.0,0.5); //etamin,1.0,etamax,1.0);
    line.SetLineColor(9);
    line.SetLineWidth(2);
    line.SetLineStyle(2);
    line.Draw("same");
    pad->Update();
  }
  if (save) {
    sprintf (name, "%s.pdf", pad->GetName());
    pad->Print(name);
  }
}

void PlotHistCorrFactors(std::string infile1, std::string text1, 
			 std::string infile2, std::string text2, 
			 bool ratio=false, bool drawStatBox=true,
			 int nmin=100, bool save=false) {

  std::vector<cfactors> cfacs1, cfacs2;
  int etamin(100), etamax(-100), maxdepth(0);
  unsigned int nhist[2];
  for (int k1=0; k1<2; ++k1) {
    std::string infile = (k1 == 0) ? infile1 : infile2;
    std::ifstream fInput(infile.c_str());
    if (!fInput.good()) {
      std::cout << "Cannot open file " << infile << std::endl;
    } else {
      char buffer [1024];
      unsigned int all(0), good(0);
      while (fInput.getline(buffer, 1024)) {
	++all;
	if (buffer [0] == '#') continue; //ignore comment
	std::vector <std::string> items = splitString (std::string (buffer));
	if (items.size () != 5) {
	  std::cout << "Ignore  line: " << buffer << std::endl;
	} else {
	  ++good;
	  int   ieta  = std::atoi (items[1].c_str());
	  int   depth = std::atoi (items[2].c_str());
	  float corrf = std::atof (items[3].c_str());
	  float dcorr = std::atof (items[4].c_str());
	  cfactors cfac(ieta,depth,corrf,dcorr);
	  if (k1 == 0) cfacs1.push_back(cfac);
	  else         cfacs2.push_back(cfac);
	  if (ieta > etamax) etamax = ieta;
	  if (ieta < etamin) etamin = ieta;
	  if (depth > maxdepth) maxdepth = depth;
	}
      }
      fInput.close();
      std::cout << "Reads total of " << all << " and " << good 
		<< " good records from " << infile << std::endl;
    }
  }
  /*
  for (unsigned int k = 0; k < cfacs.size(); ++k) 
    std::cout << "[" << k << "] " << cfacs[k].ieta << " " 
	      << cfacs[k].depth << " " << cfacs[k].corrf
	      << " " << cfacs[k].dcorr << std::endl;
  */

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if ((!ratio) && drawStatBox) {
    gStyle->SetOptStat(10);       gStyle->SetOptFit(10);
  } else {
    gStyle->SetOptStat(0);        gStyle->SetOptFit(0);
  }
  int colors[6] = {1,6,4,7,2,9};
  int mtype[6]  = {20,24,22,23,21,33};
  int nbin = etamax - etamin + 1;
  std::vector<TH1D*>  hists;
  std::vector<int>    entries;
  std::vector<double> fitr;
  char                name[100];
  double              dy(0);
  int                 fits(0);
  if (ratio) {
    for (int j=0; j<maxdepth; ++j) {
      sprintf (name, "hd%d", j+1);
      TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
      double sumNum(0), sumDen(0);
      for (unsigned int k = 0; k < cfacs1.size(); ++k) {
	int dep = cfacs1[k].depth;
	if (dep == j+1) {
	  int ieta = cfacs1[k].ieta;
	  int bin  = ieta - etamin + 1;
	  float val = cfacs1[k].corrf/cfacs2[k].corrf;
	  float dvl = val * sqrt(((cfacs1[k].dcorr*cfacs1[k].dcorr)/
				  (cfacs1[k].corrf*cfacs1[k].corrf)) +
				 ((cfacs2[k].dcorr*cfacs2[k].dcorr)/
				  (cfacs2[k].corrf*cfacs2[k].corrf)));
	  h->SetBinContent(bin,val);
	  h->SetBinError(bin,dvl);
	  sumNum += (val/(dvl*dvl));
	  sumDen += (1.0/(dvl*dvl));
	}
      }
      double fit = (sumDen > 0) ? (sumNum/sumDen) : 1.0;
      std::cout << "Fit to Pol0: " << fit << std::endl;
      h->SetLineColor(colors[j]);
      h->SetMarkerColor(colors[j]);
      h->SetMarkerStyle(mtype[0]);
      h->SetMarkerSize(0.9);
      h->GetXaxis()->SetTitle("i#eta");
      sprintf (name, "Ratio of Correction Factors (%s/%s)", text1.c_str(), text2.c_str());
      h->GetYaxis()->SetTitle(name);
      h->GetYaxis()->SetLabelOffset(0.005);
      h->GetYaxis()->SetTitleOffset(1.20);
      h->GetYaxis()->SetRangeUser(0.95,1.05);
      hists.push_back(h);
      fitr.push_back(fit);
    }
    nhist[1] = nhist[0] = hists.size();
  } else {
    for (int k1=0; k1<2; ++k1) {
      for (int j=0; j<maxdepth; ++j) {
	sprintf (name, "hd%d%d", j+1, k1);
	TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
	int nent(0);
	unsigned int nsize = (k1 == 0) ? cfacs1.size() : cfacs2.size();
	for (unsigned int k = 0; k < nsize; ++k) {
	  int dep = (k1 == 0) ? cfacs1[k].depth : cfacs2[k].depth;
	  if (dep == j+1) {
	    int ieta = (k1 == 0) ? cfacs1[k].ieta : cfacs2[k].ieta;
	    int bin  = ieta - etamin + 1;
	    float val = (k1 == 0) ? cfacs1[k].corrf : cfacs2[k].corrf;
	    float dvl = (k1 == 0) ? cfacs1[k].dcorr : cfacs2[k].dcorr;
	    h->SetBinContent(bin,val);
	    h->SetBinError(bin,dvl);
	    nent++;
	  }
	}
	if (nent > nmin) {
	  fits++;
	  if (drawStatBox) dy  += 0.025;
	  sprintf (name, "hdf%d%d", j+1, k1);
	  TF1* func   = new TF1(name, "pol0", etamin, etamax);
	  h->Fit(func,"+QWLR","");
	}
	h->SetLineColor(colors[j]);
	h->SetMarkerColor(colors[j]);
	h->SetMarkerStyle(mtype[k1]);
	h->SetMarkerSize(0.9);
	h->GetXaxis()->SetTitle("i#eta");
	h->GetYaxis()->SetTitle("Correction Factor");
	h->GetYaxis()->SetLabelOffset(0.005);
	h->GetYaxis()->SetTitleOffset(1.20);
	h->GetYaxis()->SetRangeUser(0.0,2.0);
	hists.push_back(h);
	entries.push_back(nent);
	if (drawStatBox) dy  += 0.025;
      }
      nhist[k1] = (hists.size());
      if (k1 > 0) nhist[k1] -= nhist[k1-1];
    }
  }
  TCanvas *pad = new TCanvas("CorrFactor","CorrFactor", 700, 500);
  pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
  double yh = 0.90;
  double yl = yh-0.025*hists.size()-dy-0.01;
  TLegend *legend = new TLegend(0.60, yl, 0.90, yl+0.025*hists.size());
  legend->SetFillColor(kWhite);
  for (unsigned int k=0; k<hists.size(); ++k) {
    if (k == 0) hists[k]->Draw("");
    else        hists[k]->Draw("sames");
    pad->Update();
    if (!ratio) {
      TPaveStats* st1 = (TPaveStats*)hists[k]->GetListOfFunctions()->FindObject("stats");
      int k1 = (k < nhist[0]) ? k : (k-nhist[0]);
      if (st1 != NULL) {
	dy = (entries[k] > nmin) ? 0.05 : 0.025;
	st1->SetLineColor(colors[k1]);
	st1->SetTextColor(colors[k1]);
	st1->SetY1NDC(yh-dy); st1->SetY2NDC(yh);
	st1->SetX1NDC(0.70); st1->SetX2NDC(0.90);
	yh -= dy;
      }
      std::string text = (k < nhist[0]) ? text1 : text2;
      sprintf (name, "Depth %d (%s)", k1+1, text.c_str());
    } else {
      sprintf (name, "Depth %d (Mean = %5.3f)", k+1, fitr[k]);
    }
    legend->AddEntry(hists[k],name,"lp");
  }
  legend->Draw("same");
  pad->Update();
  if (fits < 1) {
    pad->Range(0.0,0.0,1.0,1.0);
    TLine line = TLine(0.0,0.5,1.0,0.5); //etamin,1.0,etamax,1.0);
    line.SetLineColor(9);
    line.SetLineWidth(2);
    line.SetLineStyle(2);
    line.Draw("same");
    pad->Update();
  }
  if (save) {
    sprintf (name, "%s.pdf", pad->GetName());
    pad->Print(name);
  }
}

void PlotHistCorrSys(std::string infilec, int conds, std::string text, 
		     bool save=false) {

  std::map<int, cfactors> cfacs;
  char fname[100];
  sprintf(fname, "%s_cond0.txt", infilec.c_str());
  std::ifstream fInput(fname);
  int etamin(100), etamax(-100), maxdepth(0);
  unsigned int all(0), good(0);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << fname << std::endl;
  } else {
    char buffer [1024];
    while (fInput.getline(buffer, 1024)) {
      ++all;
      if (buffer [0] == '#') continue; //ignore comment
      std::vector <std::string> items = splitString (std::string (buffer));
      if (items.size () != 5) {
	std::cout << "Ignore  line: " << buffer << std::endl;
      } else {
	++good;
	int   ieta  = std::atoi (items[1].c_str());
	int   depth = std::atoi (items[2].c_str());
	float corrf = std::atof (items[3].c_str());
	float dcorr = std::atof (items[4].c_str());
	int   detId = std::atoi (items[0].c_str());
	cfacs[detId] = cfactors(ieta,depth,corrf,dcorr);
	if (ieta > etamax) etamax = ieta;
	if (ieta < etamin) etamin = ieta;
	if (depth > maxdepth) maxdepth = depth;
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records"
	      << " from " << fname << std::endl;
  }
  // There are good records from the master file
  if (good > 0) {
    // Now read the other files
    std::map<int, cfactors> errfacs;
    for (int i=0; i<conds; ++i) {
      sprintf(fname, "%s_cond%d.txt", infilec.c_str(), i+1);
      std::ifstream fInput(fname);
      if (!fInput.good()) {
	std::cout << "Cannot open file " << fname << std::endl;
      } else {
	char buffer [1024];
	unsigned int all(0), good(0);
	while (fInput.getline(buffer, 1024)) {
	  ++all;
	  if (buffer [0] == '#') continue; //ignore comment
	  std::vector <std::string> items = splitString (std::string (buffer));
	  if (items.size () != 5) {
	    std::cout << "Ignore  line: " << buffer << std::endl;
	  } else {
	    ++good;
	    float corrf = std::atof (items[3].c_str());
	    int   detId = std::atoi (items[0].c_str());
	    std::map<int, cfactors>::iterator itr = errfacs.find(detId);
	    if (itr == errfacs.end()) {
	      errfacs[detId] = cfactors(1,0,corrf,corrf*corrf);
	    } else {
	      int nent = (itr->second).ieta + 1;
	      float c1 = (itr->second).corrf + corrf;
	      float c2 = (itr->second).dcorr + (corrf*corrf);
	      errfacs[detId] = cfactors(nent,0,c1,c2);
	    }
	  }
	}
	fInput.close();
	std::cout << "Reads total of " << all << " and " << good 
		  << " good records from " << fname << std::endl;
      }
    }
    // find the RMS from the distributions
    for (std::map<int, cfactors>::iterator itr = errfacs.begin();
	 itr != errfacs.end(); ++itr) {
      int   nent = (itr->second).ieta;
      float mean = (itr->second).corrf/nent;
      float rms2 = (itr->second).dcorr/nent - (mean*mean);
      float rms  = rms2 > 0 ? sqrt(rms2) : 0;
      errfacs[itr->first] = cfactors(nent,0,mean,rms);
    }
    // Now combine the errors and plot
    int k(0);
    gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(10);         gStyle->SetOptFit(10);
    int colors[6] = {1,6,4,7,2,9};
    int mtype[6]  = {20,21,22,23,24,33};
    std::vector<TH1D*> hists;
    char               name[100];
    int                nbin = etamax - etamin + 1;
    for (int j=0; j<maxdepth; ++j) {
      sprintf (name, "hd%d", j+1);
      TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
      h->SetLineColor(colors[j]);
      h->SetMarkerColor(colors[j]);
      h->SetMarkerStyle(mtype[j]);
      h->GetXaxis()->SetTitle("i#eta");
      h->GetYaxis()->SetTitle("Correction Factor");
      h->GetYaxis()->SetLabelOffset(0.005);
      h->GetYaxis()->SetTitleOffset(1.20);
      h->GetYaxis()->SetRangeUser(0.0,2.0);
      hists.push_back(h);
    }
    for (std::map<int, cfactors>::iterator itr = cfacs.begin();
	 itr != cfacs.end(); ++itr,++k) {
      std::map<int, cfactors>::iterator itr2 = errfacs.find(itr->first);
      float mean2 = (itr2 == errfacs.end()) ? 0 : (itr2->second).corrf;
      float ersys = (itr2 == errfacs.end()) ? 0 : (itr2->second).dcorr;
      float erstt = (itr->second).dcorr;
      float ertot = sqrt(erstt*erstt+ersys*ersys);
      float mean  = (itr->second).corrf;
      int   ieta  = (itr->second).ieta;
      int   depth = (itr->second).depth;
      std::cout << "[" << k << "] " << ieta << " " << depth << " " 
		<< mean << ":" << mean2 << " " << erstt << ":" << ersys << ":"
		<< ertot << std::endl;
      int bin  = ieta - etamin + 1;
      hists[depth-1]->SetBinContent(bin,mean);
      hists[depth-1]->SetBinError(bin,ertot);
    }
    TCanvas *pad = new TCanvas("CorrFactorSys","CorrFactorSys", 700, 500);
    pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
    double yh = 0.90;
    double yl = yh-0.050*hists.size()-0.01;
    TLegend *legend = new TLegend(0.60, yl, 0.90, yl+0.025*hists.size());
    legend->SetFillColor(kWhite);
    for (unsigned int k=0; k<hists.size(); ++k) {
      if (k == 0) hists[k]->Draw("");
      else        hists[k]->Draw("sames");
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hists[k]->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetLineColor(colors[k]);
	st1->SetTextColor(colors[k]);
	st1->SetY1NDC(yh-0.025); st1->SetY2NDC(yh);
	st1->SetX1NDC(0.70); st1->SetX2NDC(0.90);
	yh -= 0.025;
      }
      sprintf (name, "Depth %d (%s)", k+1, text.c_str());
      legend->AddEntry(hists[k],name,"lp");
    }
    legend->Draw("same");
    pad->Update();
    if (save) {
      sprintf (name, "%s.pdf", pad->GetName());
      pad->Print(name);
    }
  }
}

void PlotHistCorrLumis(std::string infilec, int conds, double lumi,
		       bool save=false) {

  char fname[100];
  sprintf(fname, "%s_0.txt", infilec.c_str());
  std::ifstream fInput(fname);
  int etamin(100), etamax(-100), maxdepth(0), good(0);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << fname << std::endl;
  } else {
    char buffer [1024];
    while (fInput.getline(buffer, 1024)) {
      if (buffer [0] == '#') continue; //ignore comment
      std::vector <std::string> items = splitString (std::string (buffer));
      if (items.size () != 5) {
	std::cout << "Ignore  line: " << buffer << std::endl;
      } else {
	++good;
	int   ieta  = std::atoi (items[1].c_str());
	int   depth = std::atoi (items[2].c_str());
	if (ieta > etamax) etamax = ieta;
	if (ieta < etamin) etamin = ieta;
	if (depth > maxdepth) maxdepth = depth;
      }
    }
    fInput.close();
  }
  int  nbin = etamax - etamin + 1;
  std::cout << "Max Depth " << maxdepth << " and " << nbin << " eta bins for "
	    << etamin << ":" << etamax << std::endl;

  // There are good records from the master file
  int colors[8] = {4,2,6,7,1,9,3,5};
  int mtype[8]  = {20,21,22,23,24,25,26,27};
  if (good > 0) {
    // Now read the other files
    std::vector<TH1D*> hists;
    char               name[100];
    for (int i=0; i<conds; ++i) {
      int ih = (int)(hists.size());
      sprintf(fname, "%s_%d.txt", infilec.c_str(), i);
      std::ifstream fInput(fname);
      for (int j=0; j<maxdepth; ++j) {
	sprintf (name, "hd%d%d", j+1, i);
	TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
	h->SetLineColor(colors[j]);
	h->SetMarkerColor(colors[j]);
	h->SetMarkerStyle(mtype[i]);
	h->SetMarkerSize(0.9);
	h->GetXaxis()->SetTitle("i#eta");
	h->GetYaxis()->SetTitle("Fractional Error");
	h->GetYaxis()->SetLabelOffset(0.005);
	h->GetYaxis()->SetTitleOffset(1.20);
	h->GetYaxis()->SetRangeUser(0.0,0.10);
	hists.push_back(h);
      }
      if (!fInput.good()) {
	std::cout << "Cannot open file " << fname << std::endl;
      } else {
	char buffer [1024];
	unsigned int all(0), good(0);
	while (fInput.getline(buffer, 1024)) {
	  ++all;
	  if (buffer [0] == '#') continue; //ignore comment
	  std::vector <std::string> items = splitString (std::string (buffer));
	  if (items.size () != 5) {
	    std::cout << "Ignore  line: " << buffer << std::endl;
	  } else {
	    ++good;
	    int    ieta  = std::atoi (items[1].c_str());
	    int    depth = std::atoi (items[2].c_str());
	    double corrf = std::atof (items[3].c_str());
	    double dcorr = std::atof (items[4].c_str());
	    double value = dcorr/corrf;
	    int    bin   = ieta - etamin + 1;
	    hists[ih+depth-1]->SetBinContent(bin,value);
	    hists[ih+depth-1]->SetBinError(bin,0.0001);
	  }
	}
	fInput.close();
	std::cout << "Reads total of " << all << " and " << good 
		  << " good records from " << fname << std::endl;
      }
    }

    // Now plot the histograms
    gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);          gStyle->SetOptFit(0);
    TCanvas *pad = new TCanvas("CorrFactorErr","CorrFactorErr", 700, 500);
    pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
    double yh(0.89);
    TLegend *legend = new TLegend(0.60, yh-0.04*conds, 0.89, yh);
    legend->SetFillColor(kWhite);
    double lumic(lumi);
    for (unsigned int k=0; k<hists.size(); ++k) {
      if (k == 0) hists[k]->Draw("");
      else        hists[k]->Draw("sames");
      pad->Update();
      if (k%maxdepth == 0) {
	sprintf (name, "L = %5.2f fb^{-1}", lumic);
	legend->AddEntry(hists[k],name,"lp");
	lumic *= 0.5;
      }
    }
    legend->Draw("same");
    pad->Update();
    if (save) {
      sprintf (name, "%s.pdf", pad->GetName());
      pad->Print(name);
    }
  }
}

void PlotHistCorrRel(std::string infile1, std::string infile2,
		     std::string text1, std::string text2, bool save=false) {

  std::map<int, std::pair<cfactors,cfactors> > cfacs;
  char fname[100];
  int good1(1), etamin(100), etamax(-100), maxdepth(0);
  for (int ifile=0; ifile<2; ++ifile) {
    if (ifile == 0) {
      sprintf(fname, "%s.txt", infile1.c_str());
    } else {
      sprintf(fname, "%s.txt", infile2.c_str());
    }
    std::ifstream fInput(fname);
    unsigned int all(0), good(0);
    if (!fInput.good()) {
      std::cout << "Cannot open file " << fname << std::endl;
    } else {
      char buffer [1024];
      while (fInput.getline(buffer, 1024)) {
	++all;
	if (buffer [0] == '#') continue; //ignore comment
	std::vector <std::string> items = splitString (std::string (buffer));
	if (items.size () != 5) {
	  std::cout << "Ignore  line: " << buffer << std::endl;
	} else {
	  ++good;
	  int   ieta  = std::atoi (items[1].c_str());
	  int   depth = std::atoi (items[2].c_str());
	  float corrf = std::atof (items[3].c_str());
	  float dcorr = std::atof (items[4].c_str());
	  int   detId = std::atoi (items[0].c_str());
	  if (ifile == 0) {
	    cfactors fac1(ieta,depth,corrf,dcorr);
	    cfactors fac2(ieta,depth,0,-1);
	    cfacs[detId] = std::pair<cfactors,cfactors>(fac1,fac2);
	  } else {
	    cfactors fac1(ieta,depth,0,-1);
	    cfactors fac2(ieta,depth,corrf,dcorr);
	    std::map<int, std::pair<cfactors,cfactors> >::iterator itr = cfacs.find(detId);
	    if (itr != cfacs.end()) fac1 = (itr->second).first;
	    cfacs[detId] = std::pair<cfactors,cfactors>(fac1,fac2);
	  }
	  if (ieta > etamax) etamax = ieta;
	  if (ieta < etamin) etamin = ieta;
	  if (depth > maxdepth) maxdepth = depth;
	}
      }
      fInput.close();
      std::cout << "Reads total of " << all << " and " << good 
		<< " good records" << " from " << fname << std::endl;
    }
    good1 *= good;
  }
  // There are good records in bothe the files
  if (good1 > 0) {
    int k(0);
    gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(10);         gStyle->SetOptFit(10);
    int colors[6] = {1,6,4,7,2,9};
    int mtype[6]  = {20,21,22,23,24,33};
    std::vector<TH1D*> hists;
    char               name[100];
    int                nbin = etamax - etamin + 1;
    for (int i=0; i<2; ++i) {
      for (int j=0; j<maxdepth; ++j) {
	int j1 = (i == 0) ? j : maxdepth+j;
	sprintf (name, "hd%d%d", i, j+1);
	TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
	h->SetLineColor(colors[j1]);
	h->SetMarkerColor(colors[j1]);
	h->SetMarkerStyle(mtype[i]);
	h->GetXaxis()->SetTitle("i#eta");
	h->GetYaxis()->SetTitle("Correction Factor");
	h->GetYaxis()->SetLabelOffset(0.005);
	h->GetYaxis()->SetTitleOffset(1.20);
	h->GetYaxis()->SetRangeUser(0.0,2.0);
	hists.push_back(h);
      }
    }
    for (std::map<int, std::pair<cfactors,cfactors> >::iterator itr = cfacs.begin();
	 itr != cfacs.end(); ++itr,++k) {
      float mean1 = (itr->second).first.corrf;
      float error1= (itr->second).first.dcorr;
      float mean2 = (itr->second).second.corrf;
      float error2= (itr->second).second.dcorr;
      int   ieta  = (itr->second).first.ieta;
      int   depth = (itr->second).first.depth;
      /*
      std::cout << "[" << k << "] " << ieta << " " << depth << " " 
		<< mean1 << ":" << mean2 << " " << error1 << ":" << error2
		<< std::endl;
      */
      int bin  = ieta - etamin + 1;
      if (error1 >= 0) {
	hists[depth-1]->SetBinContent(bin,mean1);
	hists[depth-1]->SetBinError(bin,error1);
      }
      if (error2 >= 0) {
	hists[maxdepth+depth-1]->SetBinContent(bin,mean2);
	hists[maxdepth+depth-1]->SetBinError(bin,error2);
      }
    }
    TCanvas *pad = new TCanvas("CorrFactors","CorrFactors", 700, 500);
    pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
    double yh = 0.90;
    double yl = yh-0.050*hists.size()-0.01;
    TLegend *legend = new TLegend(0.60, yl, 0.90, yl+0.025*hists.size());
    legend->SetFillColor(kWhite);
    for (unsigned int k=0; k<hists.size(); ++k) {
      if (k == 0) hists[k]->Draw("");
      else        hists[k]->Draw("sames");
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hists[k]->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetLineColor(colors[k]);
	st1->SetTextColor(colors[k]);
	st1->SetY1NDC(yh-0.025); st1->SetY2NDC(yh);
	st1->SetX1NDC(0.70); st1->SetX2NDC(0.90);
	yh -= 0.025;
      }
      if (k < (unsigned int)(maxdepth)) {
	sprintf (name, "Depth %d (%s)", k+1, text1.c_str());
      } else {
	sprintf (name, "Depth %d (%s)", k-maxdepth+1, text2.c_str());
      }
      legend->AddEntry(hists[k],name,"lp");
    }
    legend->Draw("same");
    pad->Update();
    if (save) {
      sprintf (name, "%s.pdf", pad->GetName());
      pad->Print(name);
    }
  }
}
