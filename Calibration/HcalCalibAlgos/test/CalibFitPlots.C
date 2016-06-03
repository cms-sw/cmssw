//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L CalibFitPlots.C+g
//             For standard set of histograms from CalibMonitor
//  FitHistStandard(infile, outfile, prefix, mode, append, saveAll);
//             For extended set of histograms from CalibMonitor
//  FitHistExtended(infile, outfile, prefix, append);
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

void FitHistStandard(std::string infile,std::string outfile,std::string prefix,
		     int mode=11111, bool append=true, bool saveAll=false) {

  int iname[5]      = {0, 1, 2, 3, 4};
  int checkmode[5]  = {10, 1000, 1, 10000, 100};
  double xbins[9]   = {-21.0, -16.0, -12.0, -6.0, 0.0, 6.0, 12.0, 16.0, 21.0};
  double vbins[6]   = {0.0, 7.0, 10.0, 13.0, 16.0, 50.0};
  double dlbins[9]  = {0.0, 0.10, 0.20, 0.50, 1.0, 2.0, 2.5, 3.0, 10.0};
  std::string sname[4] = {"ratio","etaR", "dl1R","nvxR"};
  std::string lname[4] = {"Z", "E", "L", "V"};
  int         numb[4]  = {8, 8, 8, 5};
  bool        debug(true);

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
	    histo->GetXaxis()->SetTitle("i#eta");
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

void FitHistExtended(std::string infile,std::string outfile,std::string prefix,
		     bool append=true) {
  double xbins[43]= {-21.5,-20.5,-19.5,-18.5,-17.5,-16.5,-15.5,-14.5,-13.5,
		     -12.5,-11.5,-10.5,-9.5,-8.5,-7.5,-6.5,-5.5,-4.5,-3.5,
		     -2.5,-1.5,0.0,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,
		     11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5};
  std::string sname("ratio"), lname("Z");
  int         iname(2), numb(42);
  bool        debug(true);

  TFile      *file = new TFile(infile.c_str());
  std::vector<TH1D*> hists;
  char name[100];
  if (file != 0) {
    sprintf (name, "%s%s%d0", prefix.c_str(), sname.c_str(), iname);
    TH1D* hist0 = (TH1D*)file->FindObjectAny(name);
    bool  ok   = (hist0 != 0);
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
	TFitResultPtr Fit = histo->Fit("pol0","+QRWLS","",xbins[0],xbins[numb]);
	if (debug) std::cout << "Fit to Pol0: " << Fit->Value(0) << " +- "
			     << Fit->FitResult::Error(0) << std::endl;
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
