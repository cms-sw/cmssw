//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L CalibFitPlots.C+g
//             For standard set of histograms from CalibMonitor
//  FitHistStandard(infile, outfile, prefix, mode, type, append, saveAll);
//      Defaults: mode=11111, type=0, append=true, saveAll=false
//
//             For extended set of histograms from CalibMonitor
//  FitHistExtended(infile, outfile, prefix, numb, type, append, fiteta, iname);
//      Defaults: numb=50, type=3, append=true, fiteta=true, iname=2
//
//             For RBX dependence in sets of histograms from CalibMonitor 
//  FitHistRBX(infile, outfile, prefix, append, iname);
//      Defaults: append=true, iname=2
//
//             For plotting stored histograms from FitHist's
//  PlotHist(infile, prefix, text, modePlot, kopt, dataMC, drawStatBox, save);
//      Defaults: modePlot=0, kopt=0, dataMC=false, drawStatBox=true, save=false
//
//             For plotting several histograms in the same plot
//             (fits to different data sets for example)
//  PlotHists(infile, prefix, text, drawStatBox, save)
//      Defaults: drawStatBox=true; save=false;
//      Note prefix is common part for all histograms
//
//             For plotting on the same canvas plots with different
//             prefixes residing in the same file with approrprate text
//   PlotTwoHists(infile, prefix1, text1, prefix2, text2, text0, type, iname,
//                drawStatBox, save);
//      Defaults: type=0; iname=2; drawStatBox=true; save=false;
//      Note prefixN, textN have the same meaning as prefix and text for set N
//           text0 is the text for general title added within ()
//           type=0 plots response distributions and MPV of response vs ieta
//               =1 plots MPV of response vs RBX #
//
//             For plotting stored histograms from CalibTree
//   PlotThreeHists(infile, prefix1, text1, prefix2, text2, prefix3, text3,
//                  text0, type, drawStatBox, save);
//      Defaults: type=0; drawStatBox=true; save=false;
//      Note prefixN, textN have the same meaning as prefix and text for set N
//           text0 is the text for general title added within ()
//           type has the same meaning as in PlotTwoHists
//
//  PlotHistCorrResults(infile, text, prefix, save);
//      Defaults: save=false
//
//             For plotting correction factors
//  PlotHistCorrFactor(infile, text, prefix, scale, nmin, save);
//      Defaults: nmin=20, save=false
//
//             For plotting correction factors from 2 different runs on the
//             same canvas
//  PlotHistCorrFactors(infile1, text1, infile2, text2, ratio, drawStatBox,
//                      nmin, dataMC, year, save)
//      Defaults: drawStatBox=true, nmin=100, dataMC=false, year=2016, 
//                save=false
//
//             For plotting correction factors including systematics
//  PlotHistCorrSys(infilec, conds, text, save)
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
//  type     (int)          = defines eta binning type (see CalibMonitor)
//  append   (bool)         = Open the output in Update/Recreate mode (True)
//  fiteta   (bool)         = fit the eta dependence with pol0
//  iname    (int)          = choose the momentum bin (2: 40-60 GeV)
//  saveAll  (bool)         = Flag to save intermediate plots (False)
//  numb     (int)          = Number of eta bins (42 for -21:21)
//  text     (std::string)  = Extra text to be put in the text title
//  modePlot (int)          = Flag to plot E/p distribution (0);
//                            <E/p> as a function of ieta (1);
//                            <E/p> as a function of distance from L1 (2);
//                            <E/p> as a function of number of vertex (3);
//                            E/p for barrel, endcap and transition (4)
//  kopt     (int)          = Option in format "hdo" where each of d, o can 
//                            have a value of 0 or 1 to select or deselect. 
//                            o>0 to carry out pol0 fit, o>1 to restrict
//                            fit region between -20 & 20; d=1 to show grid;
//                            h=0,1 to show plots with 2- or 1-Gaussian fit
//  save     (bool)         = if true it saves the canvas as a pdf file
//  nmin     (int)          = minimum # of #ieta points needed to show the
//                            fitted line
//  scale    (double)       = constant scale factor applied to the factors
//  ratio    (bool)         = set to show the ratio plot (false)
//  drawStatBox (bool)      = set to show the statistical box (true)
//  year     (int)          = Year of data taking (applicable to Data)
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

Double_t doubleGauss(Double_t *x, Double_t *par) {
  double x1   = x[0]-par[1];
  double sig1 = par[2];
  double x2   = x[0]-par[4];
  double sig2 = par[5];
  double yval = (par[0]*std::exp(-0.5*(x1/sig1)*(x1/sig1)) +
		 par[3]*std::exp(-0.5*(x2/sig2)*(x2/sig2)));
  return yval;
}

TFitResultPtr functionFit(TH1D *hist, double *fitrange, double *startvalues, 
		 double *parlimitslo, double *parlimitshi) {

  char FunName[100];
  sprintf(FunName,"Fitfcn_%s",hist->GetName());
  TF1 *ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
  if (ffitold) delete ffitold;

  int npar(6);
  TF1 *ffit = new TF1(FunName,doubleGauss,fitrange[0],fitrange[1],npar);
  ffit->SetParameters(startvalues);
  ffit->SetLineColor(kBlue);
  ffit->SetParNames("Area1","Mean1","Width1","Area2","Mean2","Width2");
  for (int i=0; i<npar; i++) 
    ffit->SetParLimits(i, parlimitslo[i], parlimitshi[i]);
  TFitResultPtr Fit = hist->Fit(FunName,"QRWLS");
  return Fit;
}

std::pair<double,double> fitTwoGauss (TH1D* hist, bool debug) {
  double mean  = hist->GetMean(), rms = hist->GetRMS();
  double LowEdge  = mean - 1.0*rms;
  double HighEdge = mean + 1.0*rms;
  if (LowEdge < 0.15) LowEdge = 0.15;
  std::string option = (hist->GetEntries() > 100) ? "QRS" : "QRWLS";
  TF1 *g1    = new TF1("g1","gaus",LowEdge,HighEdge); 
  g1->SetLineColor(kGreen);
  TFitResultPtr Fit = hist->Fit(g1,option.c_str(),"");
  
  if (debug) {
    for (int k=0; k<3; ++k) 
      std::cout << "Initial Parameter[" << k << "] = " << Fit->Value(k) 
		<< " +- " << Fit->FitResult::Error(k) << std::endl;
  }
  double startvalues[6], fitrange[2], lowValue[6], highValue[6];
  startvalues[0] =     Fit->Value(0); lowValue[0] = 0.5*startvalues[0]; highValue[0] = 2.*startvalues[0];
  startvalues[1] =     Fit->Value(1); lowValue[1] = 0.5*startvalues[1]; highValue[1] = 2.*startvalues[1];
  startvalues[2] =     Fit->Value(2); lowValue[2] = 0.5*startvalues[2]; highValue[2] = 2.*startvalues[2];
  startvalues[3] = 0.1*Fit->Value(0); lowValue[3] = 0.0; highValue[3] = 10.*startvalues[3];
  startvalues[4] =     Fit->Value(1); lowValue[4] = 0.5*startvalues[4]; highValue[4] = 2.*startvalues[4];
  startvalues[5] = 2.0*Fit->Value(2); lowValue[5] = 0.5*startvalues[5]; highValue[5] = 100.*startvalues[5];
  fitrange[0] = mean - 2.0*rms; fitrange[1] = mean + 2.0*rms;
  TFitResultPtr Fitfun = functionFit(hist, fitrange, startvalues, lowValue, highValue);
  double wt1    = (Fitfun->Value(0))*(Fitfun->Value(2));
  double value1 = Fitfun->Value(1);
  double error1 = Fitfun->FitResult::Error(1); 
  double wt2    = (Fitfun->Value(3))*(Fitfun->Value(5));
  double value2 = Fitfun->Value(4);
  double error2 = Fitfun->FitResult::Error(4);
  double value  = (wt1*value1+wt2*value2)/(wt1+wt2);
  double error  = (sqrt((wt1*error1)*(wt1*error1)+(wt2*error2)*(wt2*error2))/
		   (wt1+wt2));
  std::cout << hist->GetName() << " Fit " << value << ":" << error
	    << " First  " << value1 << ":" << error1 << ":" << wt1
	    << " Second " << value2 << ":" << error2 << ":" << wt2 << std::endl;
  if (debug) {
  for (int k=0; k<6; ++k) 
    std::cout << hist->GetName() << ":Parameter[" << k << "] = " 
	      << Fitfun->Value(k) << " +- " << Fitfun->FitResult::Error(k) 
	      << std::endl;
  }
  return std::pair<double,double>(value,error);
}

std::pair<double,double> fitOneGauss (TH1D* hist, bool fitTwice, bool debug) {
  double mean     = hist->GetMean();
  double rms      = hist->GetRMS();
  double LowEdge  = ((mean-1.2*rms) < 0.5) ? 0.5 : (mean-1.2*rms);
  double HighEdge = (mean+1.2*rms);
  std::string option = (hist->GetEntries()>100) ? "QRS" : "QRWLS";
  TF1 *g1      = new TF1("g1","gaus",LowEdge,HighEdge); 
  g1->SetLineColor(kGreen);
  TFitResultPtr Fit1 = hist->Fit(g1,option.c_str(),"");
  double value = Fit1->Value(1);
  double error = Fit1->FitResult::Error(1);
  if (fitTwice) {
    LowEdge      = Fit1->Value(1) - 1.2*Fit1->Value(2);
    HighEdge     = Fit1->Value(1) + 1.2*Fit1->Value(2);
    if (LowEdge  < 0.5) LowEdge = 0.5;
    if (HighEdge > 5.0) HighEdge= 5.0;
    TFitResultPtr Fit  = hist->Fit("gaus",option.c_str(),"",LowEdge,HighEdge);
    value = Fit->Value(1);
    error = Fit->FitResult::Error(1); 
  }

  std::pair<double,double> meaner = GetMean(hist,0.2,2.0);
  if (debug) {
    std::cout << "Fit " << value << ":" << error << ":" << hist->GetMeanError()
	      << " Mean " << meaner.first << ":" << meaner.second;
  }
  double minvalue(0.30);
  if (value < minvalue || value > 2.0 || error > 0.5) {
    value = meaner.first; error = meaner.second;
  }
  if (debug) {
    std::cout << " Final " << value << ":" << error << std::endl;
  }
  return std::pair<double,double>(value,error);
}

void readCorrFactors(char* infile, double scale, std::map<int,cfactors>& cfacs,
		     int& etamin, int& etamax, int& maxdepth) {
  
  cfacs.clear();
  std::ifstream fInput(infile);
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
	cfactors cfac(ieta,depth,scale*corrf,scale*dcorr);
	int   detId = std::atoi (items[0].c_str());
	cfacs[detId] = cfactors(ieta,depth,corrf,dcorr);
	if (ieta > etamax)    etamax   = ieta;
	if (ieta < etamin)    etamin   = ieta;
	if (depth > maxdepth) maxdepth = depth;
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records"
	      << " from " << infile << std::endl;
  }
  /*
  unsigned k(0);
  std::cout << "Eta Range " << etamin << ":" << etamax << " Max Depth "
	    << maxdepth << std::endl;
  for (std::map<int,cfactors>::const_iterator itr = cfacs.begin();
       itr != cfacs.end(); ++itr, ++k)  
    std::cout << "[" << k << "] " << std::hex << itr->first << std::dec << ": "
	      << (itr->second).ieta << " "  << (itr->second).depth << " " 
	      << (itr->second).corrf << " " << (itr->second).dcorr << std::endl;
  */
}

void FitHistStandard(std::string infile,std::string outfile,std::string prefix,
		     int mode=11111, int type=0, bool append=true, 
		     bool saveAll=false) {

  int iname[5]      = {0, 1, 2, 3, 4};
  int checkmode[5]  = {10, 1000, 1, 10000, 100};
  double xbin0[9]   = {-21.0, -16.0, -12.0, -6.0, 0.0, 6.0, 12.0, 16.0, 21.0};
  double xbins[11]  = {-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0,
		       20.0, 25.0};
  double vbins[6]   = {0.0, 7.0, 10.0, 13.0, 16.0, 50.0};
  double dlbins[9]  = {0.0, 0.10, 0.20, 0.50, 1.0, 2.0, 2.5, 3.0, 10.0};
  std::string sname[4] = {"ratio","etaR", "dl1R","nvxR"};
  std::string lname[4] = {"Z", "E", "L", "V"};
  std::string xname[4] = {"i#eta", "i#eta", "d_{L1}", "# Vertex"};
  int         numb[4]  = {10, 8, 8, 5};
  bool        debug(true);

  if (type == 0) {
    numb[0] = 8;
    for (int i=0; i<9; ++i) xbins[i] = xbin0[i];
  }
  TFile      *file = new TFile(infile.c_str());
  std::vector<TH1D*> hists;
  char name[100];
  if (file != nullptr) {
    for (int m1=0; m1<4; ++m1) {
      for (int m2=0; m2<5; ++m2) {
	sprintf (name, "%s%s%d0", prefix.c_str(), sname[m1].c_str(), iname[m2]);
	TH1D* hist0 = (TH1D*)file->FindObjectAny(name);
	bool ok = ((hist0 != nullptr) && (hist0->GetEntries() > 25));
	if ((mode/checkmode[m2])%10 > 0 && ok) {
	  TH1D* histo(0);
	  sprintf (name, "%s%s%d", prefix.c_str(), lname[m1].c_str(),iname[m2]);
	  if (m1 <= 1)      histo = new TH1D(name, hist0->GetTitle(), numb[m1], xbins);
	  else if (m1 == 2) histo = new TH1D(name, hist0->GetTitle(), numb[m1], dlbins);
	  else              histo = new TH1D(name, hist0->GetTitle(), numb[m1], vbins);
	  int jmin(numb[m1]), jmax(0);
	  for (int j=0; j<=numb[m1]; ++j) {
	    sprintf (name, "%s%s%d%d", prefix.c_str(), sname[m1].c_str(), iname[m2], j);
	    TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
	    TH1D* hist  = (TH1D*)hist1->Clone();
	    double value(0), error(0);
	    if (hist->GetEntries() > 0) {
	      value = hist->GetMean(); error = hist->GetRMS();
	    }
	    if (hist->GetEntries() > 4) {
	      bool flag = (j == 0) ? true : false;
	      std::pair<double,double> meaner = fitOneGauss(hist,flag,debug);
	      value = meaner.first;    error = meaner.second;
	      if (j != 0) {
		if (j < jmin) jmin = j;
		if (j > jmax) jmax = j;
	      }
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
	    double LowEdge = histo->GetBinLowEdge(jmin);
	    double HighEdge= histo->GetBinLowEdge(jmax)+histo->GetBinWidth(jmax);
	    TFitResultPtr Fit = histo->Fit("pol0","+QRWLS","",LowEdge,HighEdge);
	    if (debug) {
	      std::cout << "Fit to Pol0: " << Fit->Value(0) << " +- "
			<< Fit->FitResult::Error(0) << std::endl;
	    }
	    histo->GetXaxis()->SetTitle(xname[m1].c_str());
	    histo->GetYaxis()->SetTitle("MPV(E_{HCAL}/(p-E_{ECAL}))");
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

void FitHistExtended(const char* infile, const char* outfile,std::string prefix,
		     int numb=50, int type=3, bool append=true, 
		     bool fiteta=true, int iname=2) {
  std::string sname("ratio"), lname("Z"), ename("etaB");
  bool        debug(false);
  double      xbins[99];
  double xbin[23] = {-23.0, -21.0, -19.0, -17.0, -15.0, -13.0, -11.0, -9.0,
		     -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0,
		     15.0, 17.0, 19.0, 21.0, 23.0};
  if (type == 2) {
    numb = 22;
    for (int k=0; k<=numb; ++k) xbins[k] = xbin[k];
  } else {
    int         neta = numb/2;
    for (int k=0; k<neta; ++k) {
      xbins[k]      = (k-neta)-0.5;
      xbins[numb-k] = (neta-k) + 0.5;
    }
    xbins[neta] = 0;
  }
  TFile      *file = new TFile(infile);
  std::vector<TH1D*> hists;
  char name[200];
  if (debug) {
    std::cout << infile << " " << file << std::endl;
  }
  if (file != nullptr) {
    sprintf (name, "%s%s%d0", prefix.c_str(), sname.c_str(), iname);
    TH1D* hist0 = (TH1D*)file->FindObjectAny(name);
    bool  ok   = (hist0 != nullptr);
    if (debug) {
      std::cout << name << " Pointer " << hist0 << " " << ok << std::endl;
    }
    if (ok) {
      TH1D* histo(0);
      if (numb > 0) {
	sprintf (name, "%s%s%d", prefix.c_str(), lname.c_str(), iname);
	histo = new TH1D(name, hist0->GetTitle(), numb, xbins);
      }
      if (hist0->GetEntries() > 10) {
	std::pair<double,double> meaner0 = fitTwoGauss(hist0, debug);
	std::pair<double,double> meaner1 = GetMean(hist0,0.2,2.0);
	if (debug) {
	  std::cout << "Fit " << meaner0.first << ":" << meaner0.second 
		    << " Mean1 " << hist0->GetMean() << ":"
		    << hist0->GetMeanError() << " Mean2 " << meaner1.first
		    << ":" << meaner1.second << std::endl;
	}
      }
      int         nv1(100), nv2(0);
      int jmin(numb), jmax(0);
      for (int j=0; j<=numb; ++j) {
	sprintf (name, "%s%s%d%d", prefix.c_str(), sname.c_str(), iname, j);
	TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
	if (debug) {
	  std::cout << "Get Histogram for " << name << " at " << hist1
		    << std::endl;
	}
	double value(0), error(0), total(0);
	if (hist1 == nullptr) {
	  value = 1.0;
	} else {
	  TH1D* hist  = (TH1D*)hist1->Clone();
	  if (hist->GetEntries() > 0) {
	    value = hist->GetMean(); error = hist->GetRMS();
	    for (int i=1; i<=hist->GetNbinsX(); ++i) 
	      total += hist->GetBinContent(i);
	  }
	  if (total > 4) {
	    if (nv1 > j) nv1 = j;
	    if (nv2 < j) nv2 = j;
	    if (j == 0) {
	      sprintf (name, "%sOne", hist1->GetName());
	      TH1D* hist2  = (TH1D*)hist1->Clone(name);
	      fitOneGauss(hist2,true,debug);
	      hists.push_back(hist2);
	      std::pair<double,double> meaner0 = fitTwoGauss(hist,debug);
	      value = meaner0.first;
	      error = meaner0.second;
	    } else {
	      std::pair<double,double> meaner = fitOneGauss(hist,false,debug);
	      value = meaner.first; error = meaner.second;
	    }
	    if (j != 0) {
	      if (j < jmin) jmin = j;
	      if (j > jmax) jmax = j;
	    }
	  }
	  hists.push_back(hist);
	}
	if (debug) {
	  std::cout << "Hist " << j << " Value " << value << " +- " << error
		    << std::endl;
	}
	if (j != 0) {
	  histo->SetBinContent(j, value);
	  histo->SetBinError(j, error);
	}
      }
      if (histo != nullptr) {
	if (histo->GetEntries() > 2 && fiteta) {
	  if (debug) {
	    std::cout << "Jmin/max " << jmin << ":" << jmax 
		      << ":" << histo->GetNbinsX() << std::endl;
	  }
	  double LowEdge = histo->GetBinLowEdge(jmin);
	  double HighEdge= histo->GetBinLowEdge(jmax)+histo->GetBinWidth(jmax);
	  TFitResultPtr Fit = histo->Fit("pol0","+QRWLS","",LowEdge,HighEdge);
	  if (debug) {
	    std::cout << "Fit to Pol0: " << Fit->Value(0) << " +- "
		      << Fit->FitResult::Error(0) << " in range "
		      << nv1 << ":" << xbins[nv1] << ":" << nv2 << ":" 
		      << xbins[nv2] << std::endl;
	  }
	  histo->GetXaxis()->SetTitle("i#eta");
	  histo->GetYaxis()->SetTitle("MPV(E_{HCAL}/(p-E_{ECAL}))");
	  histo->GetYaxis()->SetRangeUser(0.4,1.6);
	}
	hists.push_back(histo);
      } else {
	hists.push_back(hist0);
      }

      //Barrel,Endcap
      for (int j=1; j<=4; ++j) {
	sprintf (name, "%s%s%d%d", prefix.c_str(), ename.c_str(), iname, j);
	TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
	if (debug) {
	  std::cout << "Get Histogram for " << name << " at " << hist1
		    << std::endl;
	}
	if (hist1 != nullptr) {
	  TH1D* hist  = (TH1D*)hist1->Clone();
	  double value(0), error(0), total(0);
	  if (hist->GetEntries() > 0) {
	    value = hist->GetMean(); error = hist->GetRMS();
	    for (int i=1; i<=hist->GetNbinsX(); ++i) 
	      total += hist->GetBinContent(i);
	  }
	  if (total > 4) {
	    sprintf (name, "%sOne", hist1->GetName());
	    TH1D* hist2  = (TH1D*)hist1->Clone(name);
	    fitOneGauss(hist2,true,debug);
	    hists.push_back(hist2);
	    std::pair<double,double> meaner0 = fitTwoGauss(hist,debug);
	    value = meaner0.first;
	    error = meaner0.second;
	    std::pair<double,double> meaner = GetMean(hist,0.2,2.0);
	    if (debug) {
	      std::cout << "Fit " << value << ":" << error << ":" 
			<< hist->GetMeanError() << " Mean " 
			<< meaner.first << ":" << meaner.second << std::endl;
	    }
	  }
	  hists.push_back(hist);
	}
      }
    }
    TFile* theFile(0);
    if (append) {
      if (debug) {
	std::cout << "Open file " << outfile << " in append mode" << std::endl;
      }
      theFile = new TFile(outfile, "UPDATE");
    } else {
      if (debug) {
	std::cout << "Open file " << outfile << " in recreate mode" <<std::endl;
      }
      theFile = new TFile(outfile, "RECREATE");
    }

    theFile->cd();
    for (unsigned int i=0; i<hists.size(); ++i) {
      TH1D* hnew = (TH1D*)hists[i]->Clone();
      if (debug) {
	std::cout << "Write Histogram " << hnew->GetTitle() << std::endl;
      }
      hnew->Write();
    }
    theFile->Close();
    file->Close();
  }
}

void FitHistRBX(const char* infile, const char* outfile,std::string prefix,
		bool append=true, int iname=2) {
  std::string sname("RBX"), lname("R");
  int         numb(18);
  bool        debug(false);
  char        name[200];

  TFile      *file = new TFile(infile);
  std::vector<TH1D*> hists;
  sprintf (name, "%s%s%d", prefix.c_str(), lname.c_str(), iname);
  TH1D       *histo = new TH1D(name, "", numb, 0, numb);
  if (debug) {
    std::cout << infile << " " << file << std::endl;
  }
  if (file != nullptr) {
    for (int j=0; j<numb; ++j) {
      sprintf (name, "%s%s%d%d", prefix.c_str(), sname.c_str(), iname, j+1);
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (debug) {
	std::cout << "Get Histogram for " << name << " at " << hist1
		  << std::endl;
      }
      TH1D* hist  = (TH1D*)hist1->Clone();
      double value(0), error(0), total(0);
      if (hist->GetEntries() > 0) {
	value = hist->GetMean(); error = hist->GetRMS();
	for (int i=1; i<=hist->GetNbinsX(); ++i) 
	  total += hist->GetBinContent(i);
      }
      if (total > 4) {
	std::pair<double,double> meaner = fitOneGauss(hist,false,debug);
	value = meaner.first; error = meaner.second;
      }
      hists.push_back(hist);
      histo->SetBinContent(j+1, value);
      histo->SetBinError(j+1, error);
    }
    histo->GetXaxis()->SetTitle("RBX #");
    histo->GetYaxis()->SetTitle("MPV(E_{HCAL}/(p-E_{ECAL}))");
    histo->GetYaxis()->SetRangeUser(0.75,1.20);
    hists.push_back(histo);
 
    TFile* theFile(0);
    if (append) {
      if (debug) {
	std::cout << "Open file " << outfile << " in append mode" << std::endl;
      }
      theFile = new TFile(outfile, "UPDATE");
    } else {
      if (debug) {
	std::cout << "Open file " << outfile << " in recreate mode" <<std::endl;
      }
      theFile = new TFile(outfile, "RECREATE");
    }

    theFile->cd();
    for (unsigned int i=0; i<hists.size(); ++i) {
      TH1D* hnew = (TH1D*)hists[i]->Clone();
      if (debug) {
	std::cout << "Write Histogram " << hnew->GetTitle() << std::endl;
      }
      hnew->Write();
    }
    theFile->Close();
    file->Close();
  }
}

void PlotHist(const char* infile, std::string prefix, std::string text,
	      int mode=0, int kopt=0, double lumi=0, double ener=13.0,
	      bool dataMC=false, bool drawStatBox=true, bool save=false) {

  std::string name0[5] = {"ratio00","ratio10","ratio20","ratio30","ratio40"};
  std::string name1[5] = {"Z0", "Z1", "Z2", "Z3", "Z4"};
  std::string name2[5] = {"L0", "L1", "L2", "L3", "L4"};
  std::string name3[5] = {"V0", "V1", "V2", "V3", "V4"};
  std::string name4[8] = {"etaB21", "etaB22", "etaB23", "etaB24",
			  "etaB01", "etaB02", "etaB03", "etaB04"};
  std::string title[5] = {"Tracks with p = 20:30 GeV",
			  "Tracks with p = 30:40 GeV",
			  "Tracks with p = 40:60 GeV",
			  "Tracks with p = 60:100 GeV",
			  "Tracks with p = 20:100 GeV"};
  std::string title1[8] = {"Tracks with p = 40:60 GeV (Barrel)",
			   "Tracks with p = 40:60 GeV (Transition)",
			   "Tracks with p = 40:60 GeV (Endcap)",
			   "Tracks with p = 40:60 GeV",
			   "Tracks with p = 20:30 GeV (Barrel)",
			   "Tracks with p = 20:30 GeV (Transition)",
			   "Tracks with p = 20:30 GeV (Endcap)",
			   "Tracks with p = 20:30 GeV"};
  std::string xtitl[5] = {"E_{HCAL}/(p-E_{ECAL})","i#eta","d_{L1}","# Vertex",
			  "E_{HCAL}/(p-E_{ECAL})"};
  std::string ytitl[5] = {"Tracks","MPV(E_{HCAL}/(p-E_{ECAL}))",
			  "MPV(E_{HCAL}/(p-E_{ECAL}))",
			  "MPV(E_{HCAL}/(p-E_{ECAL}))","Tracks"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if (mode < 0 || mode > 4) mode = 0;
  if (drawStatBox) {
    int iopt(1110);
    if (mode != 0) iopt = 10;
    gStyle->SetOptStat(iopt);  gStyle->SetOptFit(1);
  } else {
    gStyle->SetOptStat(0);  gStyle->SetOptFit(0);
  }
  TFile      *file = new TFile(infile);
  TLine* line(0);
  char name[100], namep[100];
  int kmax = (mode == 4) ? 8 : 5;
  for (int k=0; k<kmax; ++k) {
    if (mode == 1) {
      sprintf (name, "%s%s", prefix.c_str(), name1[k].c_str());
    } else if (mode == 2) {
      sprintf (name, "%s%s", prefix.c_str(), name2[k].c_str());
    } else if (mode == 3) {
      sprintf (name, "%s%s", prefix.c_str(), name3[k].c_str());
    } else if (mode == 4) {
      if ((kopt/100)%10 == 0) 
	sprintf (name, "%s%s", prefix.c_str(), name4[k].c_str());
      else
	sprintf (name, "%s%sOne", prefix.c_str(), name4[k].c_str());
    } else {
      if ((kopt/100)%10 == 0) 
	sprintf (name, "%s%s", prefix.c_str(), name0[k].c_str());
      else
	sprintf (name, "%s%sOne", prefix.c_str(), name0[k].c_str());
    }
    TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
    if (hist1 != nullptr) {
      TH1D* hist = (TH1D*)(hist1->Clone()); 
      double p0(1);
      double ymin(0.90);
      sprintf (namep, "c_%s", name);
      TCanvas *pad = new TCanvas(namep, namep, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      if ((kopt/10)%10 > 0) gPad->SetGrid();
      hist->GetXaxis()->SetTitleSize(0.04);
      hist->GetXaxis()->SetTitle(xtitl[mode].c_str());
      hist->GetYaxis()->SetTitle(ytitl[mode].c_str());
      hist->GetYaxis()->SetLabelOffset(0.005);
      hist->GetYaxis()->SetTitleSize(0.04);
      hist->GetYaxis()->SetLabelSize(0.035);
      hist->GetYaxis()->SetTitleOffset(1.10);
      if (mode == 0 || mode == 4) {
	hist->GetXaxis()->SetRangeUser(0.0,2.5);
      } else {
	hist->GetYaxis()->SetRangeUser(0.8,1.40);
	if (kopt%10 > 0) {
	  int nbin = hist->GetNbinsX();
	  double LowEdge = (kopt%10 == 1) ? hist->GetBinLowEdge(1) : -20;
	  double HighEdge= (kopt%10 == 1) ? hist->GetBinLowEdge(nbin)+hist->GetBinWidth(nbin) : 20;
	  TFitResultPtr Fit = hist->Fit("pol0","+QRWLS","",LowEdge,HighEdge);
	  p0 = Fit->Value(0);
	}
      }
      hist->SetMarkerStyle(20);
      hist->SetMarkerColor(2);
      hist->SetLineColor(2);
      hist->Draw();
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
      if (st1 != nullptr) {
	ymin = (mode == 0 || mode == 4) ? 0.70 : 0.80; 
	st1->SetY1NDC(ymin); st1->SetY2NDC(0.90);
	st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
      }
      if (mode != 0 && mode != 4 && kopt%10 > 0) {
	double xmin = hist->GetBinLowEdge(1);
	int    nbin = hist->GetNbinsX();
	double xmax = hist->GetBinLowEdge(nbin)+hist->GetBinWidth(nbin);
	line = new TLine(xmin,p0,xmax,p0); //etamin,1.0,etamax,1.0);
	std::cout << xmin << ":" << xmax << ":" << p0 << std::endl;
	line->SetLineWidth(4);
	line->SetLineStyle(2);
	line->Draw("same");
      }
      pad->Update();
      double ymx(0.96), xmi(0.25), xmx(0.90);
      char txt[100];
      if (lumi > 0.1) {
	ymx = ymin - 0.005;
	xmi = 0.45;
	TPaveText *txt0 = new TPaveText(0.65,0.91,0.90,0.96,"blNDC");
	txt0->SetFillColor(0);
	sprintf (txt, "%4.1f TeV %5.1f fb^{-1}", ener, lumi);
	txt0->AddText(txt);
	txt0->Draw("same");
      }
      double ymi = ymx - 0.05;
      TPaveText *txt1 = new TPaveText(xmi,ymi,xmx,ymx,"blNDC");
      txt1->SetFillColor(0);
      if (text == "") {
	if (mode == 4) sprintf (txt, "%s", title1[k].c_str());
	else           sprintf (txt, "%s", title[k].c_str());
      } else {
        if (mode == 4) sprintf (txt, "%s (%s)", title1[k].c_str(),text.c_str());
	else           sprintf (txt, "%s (%s)", title[k].c_str(), text.c_str());
      }
      txt1->AddText(txt);
      txt1->Draw("same");
      double xmax = (dataMC) ? 0.33 : 0.44;
      ymi = (lumi > 0.1) ? 0.91 : 0.84;
      ymx = ymi + 0.05;
      TPaveText *txt2 = new TPaveText(0.11,ymi,xmax,ymx,"blNDC");
      txt2->SetFillColor(0);
      if (dataMC)  sprintf (txt, "CMS Preliminary");
      else         sprintf (txt, "CMS Simulation Preliminary");
      txt2->AddText(txt);
      txt2->Draw("same");
      pad->Modified();
      pad->Update();
      if (save) {
	sprintf (name, "%s.pdf",  pad->GetName());
	pad->Print(name);
      }	
    }
  }
}

void PlotHists(std::string infile, std::string prefix, std::string text,
	       bool drawStatBox=true, bool save=false) {
  int         colors[6] = {1,6,4,7,2,9};
  std::string types[6] = {"B", "C", "D", "E", "F", "G"};
  std::string names[2] = {"ratio20", "Z2"};
  std::string xtitl[2] = {"E_{HCAL}/(p-E_{ECAL})","i#eta"};
  std::string ytitl[2] = {"Tracks","MPV(E_{HCAL}/(p-E_{ECAL}))"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if (drawStatBox) gStyle->SetOptFit(10);
  else             gStyle->SetOptFit(0);

  char name[100], namep[100];
  TFile      *file = new TFile(infile.c_str());
  for (int i=0; i<2; ++i) {
    std::vector<TH1D*> hists;
    std::vector<int>   kks;
    double ymax(0.77);
    if (drawStatBox) {
      if (i == 0)  gStyle->SetOptStat(1100);
      else         gStyle->SetOptStat(10);
    } else {
      gStyle->SetOptStat(0);
      ymax = 0.89;
    }
    for (int k=0; k<6; ++k) {
      sprintf (name, "%s%s%s",prefix.c_str(),types[k].c_str(),names[i].c_str());
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (hist1 != nullptr) {
	hists.push_back((TH1D*)(hist1->Clone())); 
	kks.push_back(k);
      }
    }
    if (hists.size() > 0) {
      sprintf (namep, "c_%s%s", prefix.c_str(), names[i].c_str());
      TCanvas *pad = new TCanvas(namep, namep, 700, 500);
      TLegend *legend = new TLegend(0.44, 0.89-0.055*hists.size(), 0.69, ymax);
      legend->SetFillColor(kWhite);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      double ymax(0.90);
      double dy = (i == 0) ? 0.13 : 0.08;
      for (unsigned int jk=0; jk<hists.size(); ++jk) {
	int k = kks[jk];
	hists[jk]->GetXaxis()->SetTitle(xtitl[i].c_str());
	hists[jk]->GetYaxis()->SetTitle(ytitl[i].c_str());
	hists[jk]->GetYaxis()->SetLabelOffset(0.005);
	hists[jk]->GetYaxis()->SetLabelSize(0.035);
	hists[jk]->GetYaxis()->SetTitleOffset(1.15);
	if (i == 0) {
	  hists[jk]->GetXaxis()->SetRangeUser(0.0,2.5);
	} else  {
	  hists[jk]->GetYaxis()->SetRangeUser(0.5,2.0);
	}
	hists[jk]->SetMarkerStyle(20);
	hists[jk]->SetMarkerColor(colors[k]);
	hists[jk]->SetLineColor(colors[k]);
	if (jk == 0) hists[jk]->Draw();
	else         hists[jk]->Draw("sames");
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hists[jk]->GetListOfFunctions()->FindObject("stats");
	if (st1 != nullptr) {
	  double ymin = ymax - dy;
	  st1->SetLineColor(colors[k]);
	  st1->SetTextColor(colors[k]);
	  st1->SetY1NDC(ymin); st1->SetY2NDC(ymax);
	  st1->SetX1NDC(0.70); st1->SetX2NDC(0.90);
	  ymax = ymin;
	}
	sprintf (name, "%s%s", text.c_str(), types[k].c_str());
	legend->AddEntry(hists[jk],name,"lp");
      }
      legend->Draw("same");
      pad->Update();
      TPaveText *txt1 = new TPaveText(0.34,0.825,0.69,0.895,"blNDC");
      txt1->SetFillColor(0);
      char txt[100];
      sprintf (txt, "Tracks with p = 40:60 GeV");
      txt1->AddText(txt);
      txt1->Draw("same");
      TPaveText *txt2 = new TPaveText(0.11,0.825,0.33,0.895,"blNDC");
      txt2->SetFillColor(0);
      sprintf (txt, "CMS Preliminary");
      txt2->AddText(txt);
      txt2->Draw("same");
      if (!drawStatBox && i == 1) {
	double xmin = hists[0]->GetBinLowEdge(1);
	int    nbin = hists[0]->GetNbinsX();
	double xmax = hists[0]->GetBinLowEdge(nbin)+hists[0]->GetBinWidth(nbin);
	TLine line = TLine(xmin,1.0,xmax,1.0); //etamin,1.0,etamax,1.0);
	line.SetLineWidth(4);
	line.Draw("same");
	pad->Update();
      }
      pad->Modified();
      pad->Update();
      if (save) {
	sprintf (name, "%s.pdf", pad->GetName());
	pad->Print(name);
      }	
    }
  }
}

void PlotTwoHists(std::string infile, std::string prefix1, std::string text1,
		  std::string prefix2, std::string text2, std::string text0,
		  int type=0, int iname=2, double lumi=0, double ener=13.0,
		  int drawStatBox=0, bool save=false) {
  int         colors[2] = {2,4};
  int         numb[2]   = {4,1};
  std::string names0[4] = {"ratio00", "ratio00One", "etaB04One", "Z0"};
  std::string names1[4] = {"ratio20", "ratio20One", "etaB24One", "Z2"};
  std::string xtitl1[4] = {"E_{HCAL}/(p-E_{ECAL})","E_{HCAL}/(p-E_{ECAL})","E_{HCAL}/(p-E_{ECAL})","i#eta"};
  std::string ytitl1[4] = {"Tracks","Tracks","Tracks","MPV(E_{HCAL}/(p-E_{ECAL}))"};
  std::string names2[1] = {"R"};
  std::string xtitl2[1] = {"RBX #"};
  std::string ytitl2[1] = {"MPV(E_{HCAL}/(p-E_{ECAL}))"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if ((drawStatBox/10)%10 > 0) gStyle->SetOptFit(10);
  else                         gStyle->SetOptFit(0);

  if (type != 1) type = 0;  
  char name[100], namep[100];
  TFile      *file = new TFile(infile.c_str());
  for (int i=0; i<numb[type]; ++i) {
    std::vector<TH1D*> hists;
    std::vector<int>   kks;
    double ymax(0.77);
    if (drawStatBox%10 > 0) {
      if (i != 2)  gStyle->SetOptStat(1100);
      else         gStyle->SetOptStat(10);
    } else {
      gStyle->SetOptStat(0);
      ymax = 0.82;
    }
    for (int k=0; k<2; ++k) {
      std::string prefix = (k == 0) ? prefix1 : prefix2;
      if (type == 0) {
	if (iname == 0) sprintf (name, "%s%s",prefix.c_str(),names0[i].c_str());
	else            sprintf (name, "%s%s",prefix.c_str(),names1[i].c_str());
      } else {
	sprintf (name, "%s%s%d",prefix.c_str(),names2[i].c_str(),iname);
      }
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (hist1 != nullptr) {
	hists.push_back((TH1D*)(hist1->Clone())); 
	kks.push_back(k);
      }
    }
    if (hists.size() == 2) {
      if (type == 0) {
	if (iname == 0) sprintf (namep,"c_%s%s%s",prefix1.c_str(),prefix2.c_str(),names0[i].c_str());
	else            sprintf (namep,"c_%s%s%s",prefix1.c_str(),prefix2.c_str(),names1[i].c_str());
      } else {
	sprintf (namep,"c_%s%s%s%d",prefix1.c_str(),prefix2.c_str(),names2[i].c_str(),iname);
      }
      double ymax(0.90);
      double dy   = (i == 0) ? 0.13 : 0.08;
      double ymx0 = (drawStatBox==0) ? (ymax-.01) : (ymax-dy*hists.size()-.01);
      TCanvas *pad = new TCanvas(namep, namep, 700, 500);
      TLegend *legend = new TLegend(0.64, ymx0-0.05*hists.size(), 0.89, ymx0);
      legend->SetFillColor(kWhite);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      for (unsigned int jk=0; jk<hists.size(); ++jk) {
	int k = kks[jk];
	hists[jk]->GetXaxis()->SetTitleSize(0.040);
	if (type == 0) {
	  hists[jk]->GetXaxis()->SetTitle(xtitl1[i].c_str());
	  hists[jk]->GetYaxis()->SetTitle(ytitl1[i].c_str());
	} else {
	  hists[jk]->GetXaxis()->SetTitle(xtitl2[i].c_str());
	  hists[jk]->GetYaxis()->SetTitle(ytitl2[i].c_str());
	}
	hists[jk]->GetYaxis()->SetLabelOffset(0.005);
	hists[jk]->GetYaxis()->SetLabelSize(0.035);
	hists[jk]->GetYaxis()->SetTitleSize(0.040);
	hists[jk]->GetYaxis()->SetTitleOffset(1.15);
	if ((type == 0) && (i != 3))
	  hists[jk]->GetXaxis()->SetRangeUser(0.0,5.0);
	if ((type == 0) && (i == 3))
	  hists[jk]->GetYaxis()->SetRangeUser(0.5,1.5);
	if (type != 0)
	  hists[jk]->GetYaxis()->SetRangeUser(0.75,1.2);
	hists[jk]->SetMarkerStyle(20);
	hists[jk]->SetMarkerColor(colors[k]);
	hists[jk]->SetLineColor(colors[k]);
	if (jk == 0) hists[jk]->Draw();
	else         hists[jk]->Draw("sames");
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hists[jk]->GetListOfFunctions()->FindObject("stats");
	if (st1 != nullptr) {
	  double ymin = ymax - dy;
	  st1->SetLineColor(colors[k]);
	  st1->SetTextColor(colors[k]);
	  st1->SetY1NDC(ymin); st1->SetY2NDC(ymax);
	  st1->SetX1NDC(0.70); st1->SetX2NDC(0.90);
	  ymax = ymin;
	}
	if (k == 0) sprintf (name, "%s", text1.c_str());
	else        sprintf (name, "%s", text2.c_str());
	legend->AddEntry(hists[jk],name,"lp");
      }
      legend->Draw("same");
      pad->Update();
      char txt[100];
      double xmi(0.10), xmx(0.895), ymx(0.95);
      if (lumi > 0.01) {
	xmx = 0.70;
	xmi = 0.30;
	TPaveText *txt0 = new TPaveText(0.705,0.905,0.90,0.95,"blNDC");
	txt0->SetFillColor(0);
	sprintf (txt, "%4.1f TeV %5.1f fb^{-1}", ener, lumi);
	txt0->AddText(txt);
	txt0->Draw("same");
      }
      double ymi = ymx-0.045;
      TPaveText *txt1 = new TPaveText(xmi,ymi,xmx,ymx,"blNDC");
      txt1->SetFillColor(0);
      if (iname == 0) sprintf (txt, "p = 20:30 GeV %s", text0.c_str());
      else            sprintf (txt, "p = 40:60 GeV %s", text0.c_str());
      txt1->AddText(txt);
      txt1->Draw("same");
      ymi = (lumi > 0.1) ? 0.905 : 0.85;
      ymx = ymi + 0.045;
      TPaveText *txt2 = new TPaveText(0.105,ymi,0.295,ymx,"blNDC");
      txt2->SetFillColor(0);
      sprintf (txt, "CMS Preliminary");
      txt2->AddText(txt);
      txt2->Draw("same");
      if ((drawStatBox == 0) && (i == 2)) {
	double xmin = hists[0]->GetBinLowEdge(1);
	int    nbin = hists[0]->GetNbinsX();
	double xmax = hists[0]->GetBinLowEdge(nbin)+hists[0]->GetBinWidth(nbin);
	TLine line = TLine(xmin,1.0,xmax,1.0); //etamin,1.0,etamax,1.0);
	line.SetLineWidth(4);
	line.Draw("same");
	pad->Update();
      }
      pad->Modified();
      pad->Update();
      if (save) {
	sprintf (name, "%s.pdf", pad->GetName());
	pad->Print(name);
      }	
    }
  }
}

void PlotThreeHists(std::string infile, std::string prefix1, std::string text1,
		    std::string prefix2, std::string text2, std::string prefix3,
		    std::string text3, std::string text0, int type=0,
		    int iname=0, int drawStatBox=0, bool normalize=false,
		    bool save=false) {
  int         colors[3] = {2,4,6};
  int         numb[3]   = {4,1,4};
  std::string names0[4] = {"ratio00", "ratio00One", "etaB04", "Z0"};
  std::string names1[4] = {"ratio20", "ratio20One", "etaB24", "Z2"};
  std::string xtitl1[4] = {"E_{HCAL}/(p-E_{ECAL})","E_{HCAL}/(p-E_{ECAL})",
			   "E_{HCAL}/(p-E_{ECAL})","i#eta"};
  std::string ytitl1[4] = {"Tracks","Tracks","Tracks",
			   "MPV(E_{HCAL}/(p-E_{ECAL}))"};
  std::string names2[1] = {"R"};
  std::string xtitl2[1] = {"RBX #"};
  std::string ytitl2[1] = {"MPV(E_{HCAL}/(p-E_{ECAL}))"};
  std::string names3[4] = {"pp21","pp22","pp23","pp24"};
  std::string xtitl3[4] = {"p (GeV)", "p (GeV)", "p (GeV)", "p (GeV)"};
  std::string ytitl3[4] = {"Tracks", "Tracks", "Tracks", "Tracks"};
  std::string title3[4] = {"Barrel", "Transition", "Endcap", "Combined"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if ((drawStatBox/10)%10 > 0) gStyle->SetOptFit(10);
  else                         gStyle->SetOptFit(0);

  if (type != 1 && type != 2) type = 0;  
  char name[100], namep[100];
  TFile      *file = new TFile(infile.c_str());
  for (int i=0; i<numb[type]; ++i) {
    std::vector<TH1D*> hists;
    std::vector<int>   kks;
    double ymax(0.77);
    if (drawStatBox%10 > 0) {
      if (type == 2)   gStyle->SetOptStat(1110);
      else if (i != 3) gStyle->SetOptStat(1100);
      else             gStyle->SetOptStat(10);
    } else {
      gStyle->SetOptStat(0);
      ymax = 0.82;
    }
    for (int k=0; k<3; ++k) {
      std::string prefix = (k == 0) ? prefix1 : ((k == 1) ? prefix2 : prefix3);
      if (type == 0) {
	if (iname == 0) sprintf (name, "%s%s",prefix.c_str(),names0[i].c_str());
	else            sprintf (name, "%s%s",prefix.c_str(),names1[i].c_str());
      } else if (type == 1) {
	sprintf (name, "%s%s%d",prefix.c_str(),names2[i].c_str(),iname);
      } else {
	sprintf (name, "%s%s",prefix.c_str(),names3[i].c_str());
      }
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (hist1 != nullptr) {
	hists.push_back((TH1D*)(hist1->Clone())); 
	kks.push_back(k);
      }
    }
    if (hists.size() == 3) {
      if (type == 0) {
	if (iname == 0) sprintf (namep,"c_%s%s%s%s",prefix1.c_str(),prefix2.c_str(),prefix3.c_str(),names0[i].c_str());
	else            sprintf (namep,"c_%s%s%s%s",prefix1.c_str(),prefix2.c_str(),prefix3.c_str(),names1[i].c_str());
      } else if (type == 1) {
	sprintf (namep,"c_%s%s%s%s%d",prefix1.c_str(),prefix2.c_str(),prefix3.c_str(),names2[i].c_str(),iname);
      } else {
	sprintf (namep,"c_%s%s%s%s",prefix1.c_str(),prefix2.c_str(),prefix3.c_str(),names3[i].c_str());
      }
      double ymax(0.90);
      double dy   = (i == 0 && type == 0) ? 0.13 : 0.08;
      double ymx0 = (drawStatBox==0) ? (ymax-.01) : (ymax-dy*hists.size()-.01);
      TCanvas *pad = new TCanvas(namep, namep, 700, 500);
      TLegend *legend = new TLegend(0.64, ymx0-0.05*hists.size(), 0.89, ymx0);
      legend->SetFillColor(kWhite);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      for (unsigned int jk=0; jk<hists.size(); ++jk) {
	int k = kks[jk];
	hists[jk]->GetXaxis()->SetTitleSize(0.040);
	if (type == 0) {
	  hists[jk]->GetXaxis()->SetTitle(xtitl1[i].c_str());
	  hists[jk]->GetYaxis()->SetTitle(ytitl1[i].c_str());
	} else if (type == 1) {
	  hists[jk]->GetXaxis()->SetTitle(xtitl2[i].c_str());
	  hists[jk]->GetYaxis()->SetTitle(ytitl2[i].c_str());
	} else {
	  hists[jk]->GetXaxis()->SetTitle(xtitl3[i].c_str());
	  hists[jk]->GetYaxis()->SetTitle(ytitl3[i].c_str());
	}
	hists[jk]->GetYaxis()->SetLabelOffset(0.005);
	hists[jk]->GetYaxis()->SetLabelSize(0.035);
	hists[jk]->GetYaxis()->SetTitleSize(0.040);
	hists[jk]->GetYaxis()->SetTitleOffset(1.15);
	if ((type == 0) && (i != 3))
	  hists[jk]->GetXaxis()->SetRangeUser(0.0,2.5);
	if ((type == 0) && (i == 3))
	  hists[jk]->GetYaxis()->SetRangeUser(0.5,1.5);
	if (type == 1)
	  hists[jk]->GetYaxis()->SetRangeUser(0.75,1.2);
	hists[jk]->SetMarkerStyle(20);
	hists[jk]->SetMarkerColor(colors[k]);
	hists[jk]->SetLineColor(colors[k]);
	if (normalize && ((type == 2) || ((type == 0) && (i != 3)))) {
	  if (jk == 0) hists[jk]->DrawNormalized();
	  else         hists[jk]->DrawNormalized("sames");
	} else {
	  if (jk == 0) hists[jk]->Draw();
	  else         hists[jk]->Draw("sames");
	}
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hists[jk]->GetListOfFunctions()->FindObject("stats");
	if (st1 != nullptr) {
	  double ymin = ymax - dy;
	  st1->SetLineColor(colors[k]);
	  st1->SetTextColor(colors[k]);
	  st1->SetY1NDC(ymin); st1->SetY2NDC(ymax);
	  st1->SetX1NDC(0.70); st1->SetX2NDC(0.90);
	  ymax = ymin;
	}
	if      (k == 0) sprintf (name, "%s", text1.c_str());
	else if (k == 1) sprintf (name, "%s", text2.c_str());
	else             sprintf (name, "%s", text3.c_str());
	legend->AddEntry(hists[jk],name,"lp");
      }
      legend->Draw("same");
      pad->Update();
      TPaveText *txt1 = new TPaveText(0.10,0.905,0.80,0.95,"blNDC");
      txt1->SetFillColor(0);
      char txt[100];
      if (type == 2) {
	sprintf (txt, "p = 40:60 GeV (%s)", title3[i].c_str());
      } else if (((type == 0) && (iname == 0))) {
	sprintf (txt, "p = 20:30 GeV %s", text0.c_str());
      } else {
	sprintf (txt, "p = 40:60 GeV %s", text0.c_str());
      }
      txt1->AddText(txt);
      txt1->Draw("same");
      TPaveText *txt2 = new TPaveText(0.11,0.825,0.33,0.895,"blNDC");
      txt2->SetFillColor(0);
      sprintf (txt, "CMS Preliminary");
      txt2->AddText(txt);
      txt2->Draw("same");
      if ((drawStatBox == 0) && (i == 3)) {
	double xmin = hists[0]->GetBinLowEdge(1);
	int    nbin = hists[0]->GetNbinsX();
	double xmax = hists[0]->GetBinLowEdge(nbin)+hists[0]->GetBinWidth(nbin);
	TLine line = TLine(xmin,1.0,xmax,1.0); //etamin,1.0,etamax,1.0);
	line.SetLineWidth(4);
	line.Draw("same");
	pad->Update();
      }
      pad->Modified();
      pad->Update();
      if (save) {
	sprintf (name, "%s.pdf", pad->GetName());
	pad->Print(name);
      }	
    }
  }
}

void PlotHistCorrResults(std::string infile, std::string text, 
			 std::string prefix, bool save=false) {

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
    if (hist1 != nullptr) {
      TH1D* hist = (TH1D*)(hist1->Clone()); 
      sprintf (namep, "c_%s%s", prefix.c_str(), name[k].c_str());
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
      if (st1 != nullptr) {
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

void PlotHistCorrFactor(char* infile, std::string text, 
			std::string prefix="", double scale=1.0,
			int nmin=20, bool save=false) {

  std::map<int,cfactors> cfacs;
  int etamin(100), etamax(-100), maxdepth(0);
  readCorrFactors(infile,scale,cfacs,etamin,etamax,maxdepth);

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
    for (std::map<int,cfactors>::const_iterator itr = cfacs.begin(); 
	 itr != cfacs.end(); ++itr) {
      if ((itr->second).depth == j+1) {
	int ieta = (itr->second).ieta;
	int bin  = ieta - etamin + 1;
	float val = (itr->second).corrf;
	float dvl = (itr->second).dcorr;
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
  sprintf (name, "c_%sCorrFactor", prefix.c_str());
  TCanvas *pad = new TCanvas(name, name, 700, 500);
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
    if (st1 != nullptr) {
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

void PlotHistCorrFactors(char* infile1, std::string text1, 
			 char* infile2, std::string text2, 
			 bool ratio=false, bool drawStatBox=true,
			 int nmin=100, bool dataMC=false, int year=2016,
			 bool save=false) {

  std::map<int,cfactors> cfacs1, cfacs2;
  int etamin(100), etamax(-100), maxdepth(0);
  readCorrFactors(infile1,1.0,cfacs1,etamin,etamax,maxdepth);
  readCorrFactors(infile2,1.0,cfacs2,etamin,etamax,maxdepth);
  unsigned int nhist[2];

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
      std::map<int,cfactors>::const_iterator ktr = cfacs2.begin();
      for (std::map<int,cfactors>::const_iterator itr = cfacs1.begin(); 
	   itr != cfacs1.end(); ++itr,++ktr) {
	int dep = (itr->second).depth;
	if (dep == j+1) {
	  int ieta = (itr->second).ieta;
	  int bin  = ieta - etamin + 1;
	  float val = (itr->second).corrf/(ktr->second).corrf;
	  float dvl = val * sqrt((((itr->second).dcorr*(itr->second).dcorr)/
				  ((itr->second).corrf*(itr->second).corrf)) +
				 (((ktr->second).dcorr*(ktr->second).dcorr)/
				  ((ktr->second).corrf*(ktr->second).corrf)));
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
      sprintf (name, "CF_{%s}/CF_{%s}", text1.c_str(), text2.c_str());
      h->GetYaxis()->SetTitle(name);
      h->GetYaxis()->SetLabelOffset(0.005);
      h->GetYaxis()->SetTitleSize(0.036);
      h->GetYaxis()->SetTitleOffset(1.20);
      h->GetYaxis()->SetRangeUser(0.95,1.05);
      hists.push_back(h);
      fitr.push_back(fit);
    }
    nhist[1] = nhist[0] = hists.size();
  } else {
    std::map<int,cfactors>::const_iterator itr;
    for (int k1=0; k1<2; ++k1) {
      for (int j=0; j<maxdepth; ++j) {
	sprintf (name, "hd%d%d", j+1, k1);
	TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
	int nent(0);
	for (itr = ((k1 == 0) ? cfacs1.begin() : cfacs2.begin()); 
	     itr != ((k1 == 0) ? cfacs1.end() : cfacs2.end()); ++itr) {
	  int dep = (itr->second).depth;
	  if (dep == j+1) {
	    int ieta = (itr->second).ieta;
	    int bin  = ieta - etamin + 1;
	    float val = (itr->second).corrf;
	    float dvl = (itr->second).dcorr;
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
  if (ratio) sprintf (name, "Corr%s%sRatio", text1.c_str(), text2.c_str());
  else       sprintf (name, "Corr%s%s", text1.c_str(), text2.c_str());
  TCanvas *pad = new TCanvas(name, name, 700, 500);
  pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
  double yh = 0.90;
  double yl = yh-0.035*hists.size()-dy-0.01;
  TLegend *legend = new TLegend(0.55, yl, 0.90, yl+0.035*hists.size());
  legend->SetFillColor(kWhite);
  for (unsigned int k=0; k<hists.size(); ++k) {
    if (k == 0) hists[k]->Draw("");
    else        hists[k]->Draw("sames");
    pad->Update();
    if (!ratio) {
      TPaveStats* st1 = (TPaveStats*)hists[k]->GetListOfFunctions()->FindObject("stats");
      int k1 = (k < nhist[0]) ? k : (k-nhist[0]);
      if (st1 != nullptr) {
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
  TPaveText *txt0 = new TPaveText(0.12,0.84,0.49,0.89,"blNDC");
  txt0->SetFillColor(0);
  char txt[40];
  if (dataMC)  sprintf (txt, "CMS Preliminary (%d)", year);
  else         sprintf (txt, "CMS Simulation Preliminary (%d)", year);
  txt0->AddText(txt);
  txt0->Draw("same");
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

  char fname[100];
  sprintf(fname, "%s_cond0.txt", infilec.c_str());
  int etamin(100), etamax(-100), maxdepth(0);
  std::map<int,cfactors> cfacs;
  readCorrFactors(fname,1.0,cfacs,etamin,etamax,maxdepth);
  // There are good records from the master file
  if (cfacs.size() > 0) {
    // Now read the other files
    std::map<int, cfactors> errfacs;
    for (int i=0; i<conds; ++i) {
      sprintf(fname, "%s_cond%d.txt", infilec.c_str(), i+1);
      std::map<int,cfactors> cfacx;
      int etamin1(100), etamax1(-100), maxdepth1(0);
      readCorrFactors(fname,1.0,cfacx,etamin1,etamax1,maxdepth1);
      for (std::map<int,cfactors>::const_iterator itr1 = cfacx.begin(); 
	   itr1 != cfacx.end(); ++itr1) {
	std::map<int, cfactors>::iterator itr2 = errfacs.find(itr1->first);
	float corrf = (itr1->second).corrf;
	if (itr2 == errfacs.end()) {
	  errfacs[itr1->first] = cfactors(1,0,corrf,corrf*corrf);
	} else {
	  int nent = (itr2->second).ieta + 1;
	  float c1 = (itr2->second).corrf + corrf;
	  float c2 = (itr2->second).dcorr + (corrf*corrf);
	  errfacs[itr1->first] = cfactors(nent,0,c1,c2);
	}
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
      if (st1 != nullptr) {
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
  std::map<int,cfactors> cfacs;
  int etamin(100), etamax(-100), maxdepth(0);
  readCorrFactors(fname,1.0,cfacs,etamin,etamax,maxdepth);
  int  nbin = etamax - etamin + 1;
  std::cout << "Max Depth " << maxdepth << " and " << nbin << " eta bins for "
	    << etamin << ":" << etamax << std::endl;

  // There are good records from the master file
  int colors[8] = {4,2,6,7,1,9,3,5};
  int mtype[8]  = {20,21,22,23,24,25,26,27};
  if (cfacs.size() > 0) {
    // Now read the other files
    std::vector<TH1D*> hists;
    char               name[100];
    for (int i=0; i<conds; ++i) {
      int ih = (int)(hists.size());
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
      sprintf(fname, "%s_%d.txt", infilec.c_str(), i);
      int etamin1(100), etamax1(-100), maxdepth1(0);
      readCorrFactors(fname,1.0,cfacs,etamin1,etamax1,maxdepth1);
      for (std::map<int,cfactors>::const_iterator itr = cfacs.begin(); 
	   itr != cfacs.end(); ++itr) {
	double value = (itr->second).dcorr/(itr->second).corrf;
	int    bin   = (itr->second).ieta - etamin + 1;
	hists[ih+(itr->second).depth-1]->SetBinContent(bin,value);
	hists[ih+(itr->second).depth-1]->SetBinError(bin,0.0001);
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

void PlotHistCorrRel(char* infile1, char* infile2, std::string text1, 
		     std::string text2, bool save=false) {

  std::map<int, cfactors> cfacs1, cfacs2;
  int etamin(100), etamax(-100), maxdepth(0);
  readCorrFactors(infile1,1.0,cfacs1,etamin,etamax,maxdepth);
  readCorrFactors(infile2,1.0,cfacs2,etamin,etamax,maxdepth);
  std::map<int, std::pair<cfactors,cfactors> > cfacs;
  for (std::map<int,cfactors>::iterator itr = cfacs1.begin(); 
       itr != cfacs1.end(); ++itr) {
    std::map<int,cfactors>::iterator ktr = cfacs2.find(itr->first);
    if (ktr == cfacs2.end()) {
      cfactors fac2(((itr->second).ieta),((itr->second).depth),0,-1);
      cfacs[itr->first] = std::pair<cfactors,cfactors>((itr->second),fac2);
    } else {
      cfactors fac2(ktr->second);
      cfacs[itr->first] = std::pair<cfactors,cfactors>((itr->second),fac2);
    }
  }
  for (std::map<int,cfactors>::iterator itr = cfacs2.begin(); 
       itr != cfacs2.end(); ++itr) {
    std::map<int,cfactors>::const_iterator ktr = cfacs1.find(itr->first);
    if (ktr == cfacs1.end()) {
      cfactors fac1(((itr->second).ieta),((itr->second).depth),0,-1);
      cfacs[itr->first] = std::pair<cfactors,cfactors>(fac1,(itr->second));
    }
  }

  // There are good records in bothe the files
  if ((cfacs1.size() > 0) && (cfacs2.size() > 0)) {
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
    for (std::map<int,std::pair<cfactors,cfactors> >::iterator it=cfacs.begin();
	 it != cfacs.end(); ++it,++k) {
      float mean1 = (it->second).first.corrf;
      float error1= (it->second).first.dcorr;
      float mean2 = (it->second).second.corrf;
      float error2= (it->second).second.dcorr;
      int   ieta  = (it->second).first.ieta;
      int   depth = (it->second).first.depth;
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
      if (st1 != nullptr) {
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
