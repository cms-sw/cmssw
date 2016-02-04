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

     For the ABCD method:
     ^^^^^^^^^^^^^^^^^^^^
     you have to insert in the 2nd line instead of wenu or zee the keyword abcd(...)
     The files should contain ewk samples, sig samples and qcd samples (but also read
     later). The only absolutely necessary files are the sig ones.
     Example:
     abcd(I=0.95,dI=0.01,Fz=0.6,dFz=0.01,FzP=0.56, dFzP=0.2,ewkerror=0.1,METCut=30.,mc)
     These parameters keep the same notation as in the note. The last parameter (data)
     can take 3 values:
     data: calculate in ABCD as in data. This means that the histograms denoted with
           sig,qcd,bce,gje  are used as of the same kind and ewk as the MC ewk. 
           The background is substructed as in data
     mcOnly: here we ignore all the input parameters I, dI etc. All parameters are taken
           from MC by forcing Fqcd=1
     mc:   <THIS IS WHAT ONE NORMALLY USES> input mc samples, calculation of statistical
           and systematics as in CMS AN 2009/004, systematic and statistic error 
	   calculation. This option also creates the plots of the variation of the
           signal prediction vs the parameter variation. In order to set the limits of
           the desired variation you have to edit the values in line 113 of this code
           (they are hardwired in the code)
     TO DO:
     functionalities to plot more kind of plots, e.g. efficiencies
     
     
     Further Questions/Contact:
     
         nikolaos.rompotis @ cern.ch



	 Nikolaos Rompotis - 29 June 09
	 18 Sept 09:  1st updgrade: input files in a text file
	 28 May  10:  bug in IMET corrected, thanks to Sadia Khalil
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
#include "TGraph.h"
#include "TLegend.h"

void plotMaker(TString histoName, TString typeOfplot,
	       vector<TString> file, vector<TString> type, 
	       vector<double> weight,  TString xtitle);

void abcd(vector<TString> file, vector<TString> type, vector<double> weight,
	  double METCut, double I, double dI, double Fz, double dFz, 
	  double FzP, double dFzP, double ewkerror,
	  double data, double mc, double mcOnly);
double  searchABCDstring(TString abcdString, TString keyword);
double Trionym(double a, double b, double c, double sum);
double CalcABCD
(double I, double Fz, double FzP, double K, double ewk,
 double Na_, double Nb_, double Nc_, double Nd_,
 double Ea_, double Eb_, double Ec_, double Ed_);

// values for systematics plots: it is fraction of the MC value
const double EWK_SYST_MIN = 0.3;
const double EWK_SYST_MAX = 0.3;
//
const double I_SYST_MIN = 0.05;
const double I_SYST_MAX = 0.05;
//
const double FZ_SYST_MIN = 0.1;
const double FZ_SYST_MAX = 0.1;
//
const double FZP_SYST_MIN = 0.1;
const double FZP_SYST_MAX = 0.1;
//
const double K_SYST_MIN = 0.8;
const double K_SYST_MAX = 0.8;


using namespace std;

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
	  if (w>0)
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
    plotMaker("h_met", typeOfplot, files, types, weights, "MET (GeV)");
  }
  else if (typeOfplot == "zee"){
    cout << "zee plot maker" << endl;
    //        ====================
    // =====> WHICH HISTOS TO PLOT
    //        ====================
    plotMaker("h_mee", typeOfplot, files, types, weights, "M_{ee} (GeV)");
  }
  else if (typeOfplot(0,4) == "abcd") {
    // now read the parameters of the ABCD method
    // look for parameter I and dI
    double I = searchABCDstring(typeOfplot, "I");
    double dI= searchABCDstring(typeOfplot, "dI");
    // look for parameter Fz
    double Fz = searchABCDstring(typeOfplot, "Fz");
    double dFz= searchABCDstring(typeOfplot, "dFz");
    // look for parameter FzP
    double FzP = searchABCDstring(typeOfplot, "FzP");
    double dFzP= searchABCDstring(typeOfplot, "dFzP");
    // look for the MET cut
    double METCut =searchABCDstring(typeOfplot, "METCut");
    // do you want data driven only?
    double data = searchABCDstring(typeOfplot, "data");
    double mc = searchABCDstring(typeOfplot, "mc");
    double mcOnly = searchABCDstring(typeOfplot, "mcOnly");
    // what is the ewk error?
    double ewkerror = searchABCDstring(typeOfplot, "ewkerror");
    // sanity check:
    if (METCut<0 || (data<-0.7 && mc<-0.7 && mcOnly<-0.7)) {
      cout << "Error in your configurtion!" << endl;
      if (METCut <0) cout << "Error in MET Cut" << endl;
      else cout << "You need to specify one mc or data or mcOnly"
		<< endl;
      abort();
    }
    if (mc>-0.7 && mc <0 && ewkerror<0) {
      cout << "You have specified mc option, but you have forgotten"
	   << " to set the ewkerror!" << endl;
      abort();
    }
    //        ===============================
    // =====> ABCD METHOD FOR BKG SUBTRACTION
    //        ===============================
    cout << "doing ABCD with input: " << typeOfplot << endl;
    abcd(files, types, weights, METCut, I, dI, Fz, dFz, FzP, dFzP,
	 ewkerror, data, mc, mcOnly);

  }
  // force the program to abort in order to clear the memory
  // and avoid further use of the interpreter after
  abort();

}

void abcd( vector<TString> file, vector<TString> type, vector<double> weight, 
	   double METCut, double I, double dI, double Fz, double dFz, 
	   double FzP, double dFzP, double ewkerror,
	   double data, double mc, double mcOnly)
{
  gROOT->Reset();
  gROOT->ProcessLine(".L tdrstyle.C"); 
  gROOT->ProcessLine("setTDRStyle()");
  //
  std::cout << "Trying ABCD method for Background subtration" << std::endl;
  //
  // histogram names to use:
  TString histoName_Ba("h_met_EB");
  TString histoName_Bb("h_met_inverse_EB");
  TString histoName_Ea("h_met_EE");
  TString histoName_Eb("h_met_inverse_EE");
  //
  // find one file and get the dimensions of your histogram
  int fmax = (int) file.size();
  int NBins = 0; double min = 0; double max = -1;
  for (int i=0; i<fmax; ++i) {
    if (weight[i]>0) {
      //      cout << "Loading file " << file[i] << endl;
      TFile f(file[i]);
      TH1F *h = (TH1F*) f.Get(histoName_Ba);
      NBins = h->GetNbinsX();
      min = h->GetBinLowEdge(1);
      max = h->GetBinLowEdge(NBins+1);
      break;
    }
  }
  if (NBins ==0 || (max<min)) {
    std::cout << "PlotCombiner::abcd error: Could not find valid histograms in file"
	      << std::endl;
    abort();
  }
  cout << "Histograms with "<< NBins <<" bins  and range " << min << "-" << max  << endl;
  //
  // Wenu Signal .......................................................
  TH1F h_wenu("h_wenu", "h_wenu", NBins, min, max);
  TH1F h_wenu_inv("h_wenu_inv", "h_wenu_inv", NBins, min, max);
  for (int i=0; i<fmax; ++i) {
    if (type[i] == "sig" && weight[i]>0 ) {
      TFile f(file[i]);
      //
      TH1F *h_ba = (TH1F*) f.Get(histoName_Ba);
      h_wenu.Add(h_ba, weight[i]);
      TH1F *h_ea = (TH1F*) f.Get(histoName_Ea);
      h_wenu.Add(h_ea, weight[i]);
      //
      TH1F *h_bb = (TH1F*) f.Get(histoName_Bb);
      h_wenu_inv.Add(h_bb, weight[i]);
      TH1F *h_eb = (TH1F*) f.Get(histoName_Eb);
      h_wenu_inv.Add(h_eb, weight[i]);
    }
  }
  // QCD Bkgs
  TH1F h_qcd("h_qcd", "h_qcd", NBins, min, max);
  TH1F h_qcd_inv("h_qcd_inv", "h_qcd_inv", NBins, min, max);
  for (int i=0; i<fmax; ++i) {
    if ((type[i] == "qcd" || type[i] == "bce" 
	 || type[i] == "gje") && weight[i]>0) {
      TFile f(file[i]);
      //
      TH1F *h_ba = (TH1F*) f.Get(histoName_Ba);
      h_qcd.Add(h_ba, weight[i]);
      TH1F *h_ea = (TH1F*) f.Get(histoName_Ea);
      h_qcd.Add(h_ea, weight[i]);
      //
      TH1F *h_bb = (TH1F*) f.Get(histoName_Bb);
      h_qcd_inv.Add(h_bb, weight[i]);
      TH1F *h_eb = (TH1F*) f.Get(histoName_Eb);
      h_qcd_inv.Add(h_eb, weight[i]);
    }
  }
  //
  TH1F h_ewk("h_ewk", "h_ewk", NBins, min, max);
  TH1F h_ewk_inv("h_ewk_inv", "h_ewk_inv", NBins, min, max);
  for (int i=0; i<fmax; ++i) {
    if ( type[i] == "ewk" && weight[i]>0) {
      TFile f(file[i]);
      //
      TH1F *h_ba = (TH1F*) f.Get(histoName_Ba);
      h_ewk.Add(h_ba, weight[i]);
      TH1F *h_ea = (TH1F*) f.Get(histoName_Ea);
      h_ewk.Add(h_ea, weight[i]);
      //
      TH1F *h_bb = (TH1F*) f.Get(histoName_Bb);
      h_ewk_inv.Add(h_bb, weight[i]);
      TH1F *h_eb = (TH1F*) f.Get(histoName_Eb);
      h_ewk_inv.Add(h_eb, weight[i]);
    }
  }
  //
  // calculate the METCut position
  //
  // this is calculated as a low edge bin of your input histogram
  // METCut = min + (max-min)*IMET/NBins
  int IMET = int ((METCut - min)/(max-min) * double(NBins)); 
  // check whether it is indeed a low egde position
  double metCalc = min + (max-min)*double(IMET)/double(NBins);
  if (metCalc < METCut || metCalc > METCut) {
    std::cout << "PlotCombiner:abcd: your MET Cut is not in low egde bin position"
	      << std::endl;
  }
  cout << "MET Cut in " << METCut << "GeV corresponds to bin #" << IMET << endl;
  // Calculate the population in the ABCD Regions now
  // signal
  double a_sig = h_wenu.Integral(IMET,NBins+1);
  double b_sig = h_wenu.Integral(0,IMET-1);
  double c_sig = h_wenu_inv.Integral(0,IMET-1);
  double d_sig = h_wenu_inv.Integral(IMET,NBins+1);
  // qcd
  double a_qcd = h_qcd.Integral(IMET,NBins+1);
  double b_qcd = h_qcd.Integral(0,IMET-1);
  double c_qcd = h_qcd_inv.Integral(0,IMET-1);
  double d_qcd = h_qcd_inv.Integral(IMET,NBins+1);
  // ewk
  double a_ewk = h_ewk.Integral(IMET,NBins+1);
  double b_ewk = h_ewk.Integral(0,IMET-1);
  double c_ewk = h_ewk_inv.Integral(0,IMET-1);
  double d_ewk = h_ewk_inv.Integral(IMET,NBins+1);
  ////////////////////////////////////////////////

  //
  // now the parameters of the method
  if (data < 0 && data >-0.75) {  // select value -0.5 that gives the
                                  // string parser
    // now everything is done from data + input
    std::cout << "Calculating ABCD Result and Stat Error Assuming DATA"
	      << std::endl << "Summary: in this implementation we have assumed"
	      << " that what real 'data' appear with type sig in the input"
	      << std::endl << "No systematics available with this type of"
	      << " calculation. If you need systematics try one of the other"
	      << " options" << std::endl;
    double A = (1.0-I)*(FzP-Fz);
    double B = I*(FzP+1.0)*(FzP*(c_sig-c_ewk)-(d_sig-d_ewk)) + 
      (1+Fz)*(1-I)*((a_sig-a_ewk)-dFzP*(b_sig-b_ewk));
    double C = I*(1.+Fz)*(1.+FzP)*((d_sig-d_ewk)*(b_sig-b_ewk) - (a_sig-a_ewk)*(c_sig-c_ewk));
    //
    // signal calculation:
    double S = Trionym(A,B,C, a_sig+b_sig);

    // the errors now:
    // calculate the statistical error now:
    double  ApI=0, ApFz=0, ApFzP=0, ApNa=0, ApNb=0, ApNc=0, ApNd=0;
    double  BpI=0, BpFz=0, BpFzP=0, BpNa=0, BpNb=0, BpNc=0, BpNd=0;
    double  CpI=0, CpFz=0, CpFzP=0, CpNa=0, CpNb=0, CpNc=0, CpNd=0;
    double  SpI=0, SpFz=0, SpFzP=0, SpNa=0, SpNb=0, SpNc=0, SpNd=0;
    //
    double Na = a_sig, Nb = b_sig, Nc=c_sig, Nd = d_sig;
    double Ea = a_ewk, Eb = b_ewk, Ec=c_ewk, Ed = d_ewk;
    if (A != 0) {
      
      ApI   = -(FzP-Fz);
      ApFz  = -(1.0-I);
      ApFzP = (1.0-I);
      ApNa  = 0.0;
      ApNb  = 0.0;
      ApNc  = 0.0;
      ApNd  = 0.0;
      
      BpI   = (FzP+1.0)*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0+Fz)*((Na-Ea)-FzP*(Nb-Eb));
      BpFz  = I*(FzP+1.0)*(Nc-Ec)+(1.0-I)*((Na-Ea)-FzP*(Nb-Eb));
      BpFzP = I*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0-I)*(1.0+Fz)*(Nb-Eb);
      BpNa  =  (1.0-I)*(1.0+Fz);
      BpNb  = -(1.0-I)*(1.0+Fz)*FzP;
      BpNc  = I*(FzP+1.0)*Fz;
      BpNd  = -I*(FzP+1.0);
      
      CpI   = (1.0+Fz)*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec)); 
      CpFz  = I*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpFzP = I*(1.0+Fz)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpNa  = -I*(1.0+Fz)*(1.0+FzP)*(Nc-Ec);
      CpNb  =  I*(1.0+Fz)*(1.0+FzP)*(Nd-Ed);
      CpNc  = -I*(1.0+Fz)*(1.0+FzP)*(Na-Ea);
      CpNd  =  I*(1.0+Fz)*(1.0+FzP)*(Nb-Eb);
      
      SpI   = (-BpI   + (B*BpI   -2.0*ApI*C   -2.0*A*CpI)  /fabs(2.0*A*S+B)- 2.0*ApI*S)  /(2.0*A);
      SpFz  = (-BpFz  + (B*BpFz  -2.0*ApFz*C  -2.0*A*CpFz) /fabs(2.0*A*S+B)- 2.0*ApFz*S) /(2.0*A);
      SpFzP = (-BpFzP + (B*BpFzP -2.0*ApFzP*C -2.0*A*CpFzP)/fabs(2.0*A*S+B)- 2.0*ApFzP*S)/(2.0*A);
      SpNa  = (-BpNa  + (B*BpNa  -2.0*ApNa*C  -2.0*A*CpNa) /fabs(2.0*A*S+B)- 2.0*ApNa*S) /(2.0*A);
      SpNb  = (-BpNb  + (B*BpNb  -2.0*ApNb*C  -2.0*A*CpNb) /fabs(2.0*A*S+B)- 2.0*ApNb*S) /(2.0*A);
      SpNc  = (-BpNc  + (B*BpNc  -2.0*ApNc*C  -2.0*A*CpNc) /fabs(2.0*A*S+B)- 2.0*ApNc*S) /(2.0*A);
      SpNd  = (-BpNd  + (B*BpNd  -2.0*ApNd*C  -2.0*A*CpNd) /fabs(2.0*A*S+B)- 2.0*ApNd*S) /(2.0*A);
    }
    else {
      BpI   = (FzP+1.0)*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0+Fz)*((Na-Ea)-FzP*(Nb-Eb));
      BpFz  = I*(FzP+1.0)*(Nc-Ec)+(1.0-I)*((Na-Ea)-FzP*(Nb-Eb));
      BpFzP = I*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0-I)*(1.0+Fz)*(Nb-Eb);
      BpNa  =  (1.0-I)*(1.0+Fz);
      BpNb  = -(1.0-I)*(1.0+Fz)*FzP;
      BpNc  = I*(FzP+1.0)*Fz;
      BpNd  = -I*(FzP+1.0);
      
      CpI   = (1.0+Fz)*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec)); 
      CpFz  = I*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpFzP = I*(1.0+Fz)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpNa  = -I*(1.0+Fz)*(1.0+FzP)*(Nc-Ec);
      CpNb  =  I*(1.0+Fz)*(1.0+FzP)*(Nd-Ed);
      CpNc  = -I*(1.0+Fz)*(1.0+FzP)*(Na-Ea);
      CpNd  =  I*(1.0+Fz)*(1.0+FzP)*(Nb-Eb);
      
      SpI   = -CpI/B+C*BpI/(B*B);
      SpFz  = -CpFz/B+C*BpFz/(B*B);
      SpFzP = -CpFzP/B+C*BpFzP/(B*B);
      SpNa  = -CpNa/B+C*BpNa/(B*B);
      SpNb  = -CpNb/B+C*BpNb/(B*B);
      SpNc  = -CpNc/B+C*BpNc/(B*B);
      SpNd  = -CpNd/B+C*BpNd/(B*B);
    }
    double  DS;
    DS = sqrt( SpI*dI*SpI*dI + SpFz*dFz*SpFz*dFz + SpFzP*dFzP*SpFzP*dFzP  +
	       SpNa*SpNa*Na + SpNb*SpNb*Nb + SpNc*SpNc*Nc + SpNd*SpNd*Nd );
    // warning: S here denotes the method prediction ..........
    cout << "********************************************************" << endl;
    cout << "Signal Prediction: " << S << "+-" << DS << "(stat)" << endl;
    cout << "********************************************************" << endl;
    cout << "Parameters used in calculation: " << endl;
    cout << "I=  " << I << "+-" << dI << endl;
    cout << "Fz= " << Fz << "+-" << dFz << endl;
    cout << "FzP=" << FzP << "+-" << dFzP << endl;
    cout << endl;
    cout << "ABCD Regions population:" << endl;
    cout << "A:  N=" << Na << ", sig=" << a_sig << ", qcd=" << a_qcd
         << ", ewk=" << a_ewk << endl;
    cout << "B:  N=" << Nb << ", sig=" << b_sig << ", qcd=" << b_qcd
         << ", ewk=" << b_ewk << endl;
    cout << "C:  N=" << Nc << ", sig=" << c_sig << ", qcd=" << c_qcd
         << ", ewk=" << c_ewk << endl;
    cout << "D:  N=" << Nd << ", sig=" << d_sig << ", qcd=" << d_qcd
         << ", ewk=" << d_ewk << endl;
    cout << endl;
    //
    cout << "Statistical Error Summary: " << endl;
    cout << "due to Fz = "<< SpFz*dFz<< ", ("<<SpFz*dFz*100./S << "%)"<< endl;
    cout << "due to FzP= "<< SpFzP*dFzP
	 << ", ("<<SpFzP*dFzP*100./S << "%)"<< endl; 
    cout << "due to  I = "<< SpI*dI
	 << ", ("<<SpI*dI*100./S << "%)"<< endl; 
    cout << "due to Na = "<< SpNa*sqrt(Na)
	 << ", ("<< SpNa*sqrt(Na)*100./S << "%)"<< endl; 
    cout << "due to Nb = "<< SpNb*sqrt(Nb)
	 << ", ("<< SpNb*sqrt(Nb)*100./S << "%)"<< endl; 
    cout << "due to Nc = "<< SpNc*sqrt(Nc)
	 << ", ("<< SpNc*sqrt(Nc)*100./S << "%)"<< endl; 
    cout << "due to Nd = "<< SpNd*sqrt(Nd)
	 << ", ("<< SpNd*sqrt(Nd)*100./S << "%)"<< endl; 
    cout << "Total Statistical Error: " 
	 << DS << ", (" << DS*100./S << "%)"<< endl;
    cout << "Stat Error percentages are wrt S prediction, not S mc" << endl;
  }
  //
  //
  //  this is the main option of the algorithm: the one implemented in the 
  //  Analysis Note
  //
  if (mc < 0 && mc >-0.75) {  // select value -0.5 that gives the 
                              // string parser
    
    //////// STATISTICAL ERROR CALCULATION /////////////////////////////
    double A = (1.0-I)*(FzP-Fz);
    double B = I*(FzP+1.0)*(FzP*(c_sig+c_qcd)-(d_sig+d_qcd)) + 
      (1+Fz)*(1-I)*((a_sig+a_qcd)-dFzP*(b_sig+b_qcd));
    double C = I*(1.+Fz)*(1.+FzP)*((d_sig+d_qcd)*(b_sig+b_qcd) - 
				   (a_sig+a_qcd)*(c_sig+c_qcd));
    //
    // signal calculation:
    double S = Trionym(A,B,C, a_sig+b_sig);
    //
    double  ApI=0, ApFz=0, ApFzP=0, ApNa=0, ApNb=0, ApNc=0, ApNd=0;
    double  BpI=0, BpFz=0, BpFzP=0, BpNa=0, BpNb=0, BpNc=0, BpNd=0;
    double  CpI=0, CpFz=0, CpFzP=0, CpNa=0, CpNb=0, CpNc=0, CpNd=0;
    double  SpI=0, SpFz=0, SpFzP=0, SpNa=0, SpNb=0, SpNc=0, SpNd=0;
    //
    double Na = a_sig+a_qcd+a_ewk, Nb = b_sig+b_qcd+b_ewk;
    double Nc=c_sig+c_qcd+c_ewk,   Nd = d_sig+d_qcd+d_ewk;
    double Ea = a_ewk, Eb = b_ewk, Ec=c_ewk, Ed = d_ewk;
    if (A != 0) {
      
      ApI   = -(FzP-Fz);
      ApFz  = -(1.0-I);
      ApFzP = (1.0-I);
      ApNa  = 0.0;
      ApNb  = 0.0;
      ApNc  = 0.0;
      ApNd  = 0.0;
      
      BpI   = (FzP+1.0)*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0+Fz)*((Na-Ea)-FzP*(Nb-Eb));
      BpFz  = I*(FzP+1.0)*(Nc-Ec)+(1.0-I)*((Na-Ea)-FzP*(Nb-Eb));
      BpFzP = I*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0-I)*(1.0+Fz)*(Nb-Eb);
      BpNa  =  (1.0-I)*(1.0+Fz);
      BpNb  = -(1.0-I)*(1.0+Fz)*FzP;
      BpNc  = I*(FzP+1.0)*Fz;
      BpNd  = -I*(FzP+1.0);
      
      CpI   = (1.0+Fz)*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec)); 
      CpFz  = I*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpFzP = I*(1.0+Fz)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpNa  = -I*(1.0+Fz)*(1.0+FzP)*(Nc-Ec);
      CpNb  =  I*(1.0+Fz)*(1.0+FzP)*(Nd-Ed);
      CpNc  = -I*(1.0+Fz)*(1.0+FzP)*(Na-Ea);
      CpNd  =  I*(1.0+Fz)*(1.0+FzP)*(Nb-Eb);
      
      SpI   = (-BpI   + (B*BpI   -2.0*ApI*C   -2.0*A*CpI)  /fabs(2.0*A*S+B)- 2.0*ApI*S)  /(2.0*A);
      SpFz  = (-BpFz  + (B*BpFz  -2.0*ApFz*C  -2.0*A*CpFz) /fabs(2.0*A*S+B)- 2.0*ApFz*S) /(2.0*A);
      SpFzP = (-BpFzP + (B*BpFzP -2.0*ApFzP*C -2.0*A*CpFzP)/fabs(2.0*A*S+B)- 2.0*ApFzP*S)/(2.0*A);
      SpNa  = (-BpNa  + (B*BpNa  -2.0*ApNa*C  -2.0*A*CpNa) /fabs(2.0*A*S+B)- 2.0*ApNa*S) /(2.0*A);
      SpNb  = (-BpNb  + (B*BpNb  -2.0*ApNb*C  -2.0*A*CpNb) /fabs(2.0*A*S+B)- 2.0*ApNb*S) /(2.0*A);
      SpNc  = (-BpNc  + (B*BpNc  -2.0*ApNc*C  -2.0*A*CpNc) /fabs(2.0*A*S+B)- 2.0*ApNc*S) /(2.0*A);
      SpNd  = (-BpNd  + (B*BpNd  -2.0*ApNd*C  -2.0*A*CpNd) /fabs(2.0*A*S+B)- 2.0*ApNd*S) /(2.0*A);
    }
    else {
      BpI   = (FzP+1.0)*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0+Fz)*((Na-Ea)-FzP*(Nb-Eb));
      BpFz  = I*(FzP+1.0)*(Nc-Ec)+(1.0-I)*((Na-Ea)-FzP*(Nb-Eb));
      BpFzP = I*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0-I)*(1.0+Fz)*(Nb-Eb);
      BpNa  =  (1.0-I)*(1.0+Fz);
      BpNb  = -(1.0-I)*(1.0+Fz)*FzP;
      BpNc  = I*(FzP+1.0)*Fz;
      BpNd  = -I*(FzP+1.0);
      
      CpI   = (1.0+Fz)*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec)); 
      CpFz  = I*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpFzP = I*(1.0+Fz)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpNa  = -I*(1.0+Fz)*(1.0+FzP)*(Nc-Ec);
      CpNb  =  I*(1.0+Fz)*(1.0+FzP)*(Nd-Ed);
      CpNc  = -I*(1.0+Fz)*(1.0+FzP)*(Na-Ea);
      CpNd  =  I*(1.0+Fz)*(1.0+FzP)*(Nb-Eb);
      
      SpI   = -CpI/B+C*BpI/(B*B);
      SpFz  = -CpFz/B+C*BpFz/(B*B);
      SpFzP = -CpFzP/B+C*BpFzP/(B*B);
      SpNa  = -CpNa/B+C*BpNa/(B*B);
      SpNb  = -CpNb/B+C*BpNb/(B*B);
      SpNc  = -CpNc/B+C*BpNc/(B*B);
      SpNd  = -CpNd/B+C*BpNd/(B*B);
    }
    double  DS;
    DS = sqrt( SpI*dI*SpI*dI + SpFz*dFz*SpFz*dFz + SpFzP*dFzP*SpFzP*dFzP  +
	       SpNa*SpNa*Na + SpNb*SpNb*Nb + SpNc*SpNc*Nc + SpNd*SpNd*Nd );

    ////////////////////////////////////////////////////////////////////////
    // SYSTEMATICS CALCULATION /////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    // recalculate the basic quantities
    double Imc = (a_sig + b_sig) / (a_sig + b_sig + c_sig + d_sig);
    double dImc = sqrt(Imc*(1-Imc)/(a_sig + b_sig + c_sig + d_sig));
    double Fzmc = a_sig/b_sig;
    double e =a_sig/(a_sig + b_sig);
    double de = sqrt(e*(1-e)/(a_sig + b_sig));
    double alpha = de/(2.*Fzmc-e);
    double dFzmc = alpha/(1-alpha);
    double FzPmc = d_sig/c_sig;
    double ep =d_sig/(c_sig + d_sig);
    double dep = sqrt(ep*(1-ep)/(c_sig + d_sig));
    double alphap = dep/(2.*FzPmc-ep);
    double dFzPmc = alphap/(1-alphap);
    //
    // calculate the K parameter as it is in MC:
    double KMC = (d_qcd/c_qcd)/(a_qcd/b_qcd);
    double SMC = a_sig + b_sig;
    //
    double dfz = Fz -Fzmc;
    double di = I - Imc;
    double dfzp = FzP - FzPmc;
    double fk = fabs(1-KMC);
    ////////////////////////////////////////////////////////////////////////
    // ewk error: this error has to be inserted by hand
    double fm = 1.-ewkerror;
    double fp = 1.+ewkerror;
    double S_EWK_PLUS = CalcABCD(Imc, Fzmc, FzPmc, KMC, fp, Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
    double S_EWK_MINUS = CalcABCD(Imc, Fzmc, FzPmc, KMC, fm, Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
    // error in K
    double S_K= CalcABCD(Imc, Fzmc, FzPmc, 1., 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
    // error in Fz
    double S_FZ= CalcABCD(Imc, Fz, FzPmc, KMC, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
    // error in FzP
    double S_FZP= CalcABCD(Imc, Fzmc, FzP, KMC, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
    // error in I
    double S_I = CalcABCD(I, Fzmc, FzPmc, KMC, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
    //
    // sanity tets
    //cout << "Smc=" << SMC<< ", " << CalcABCD(Imc, Fzmc, FzPmc, KMC, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed)
    // << endl;
    //abort();
    //
    // ************ plots for the systematics calculation ****************
    // ewk plot
    int const POINTS = 10;
    int const allPOINTS = 2*POINTS;
    TGraph g_ewk(allPOINTS);
    TGraph g_fz(allPOINTS);
    TGraph g_fzp(allPOINTS);
    TGraph g_k(allPOINTS);
    TGraph g_i(allPOINTS);
    //
    double ewk_syst_min = EWK_SYST_MIN; // because this is just fraction
    double i_syst_min = Imc*(1.-I_SYST_MIN);
    double fz_syst_min = Fzmc*(1.-FZ_SYST_MIN);
    double fzp_syst_min = FzPmc*(1.-FZP_SYST_MIN);
    double k_syst_min = KMC*(1.-K_SYST_MIN);
    //
    double ewk_syst_max = EWK_SYST_MAX; // because this is just fraction
    double i_syst_max = Imc*I_SYST_MAX;
    double fz_syst_max = Fzmc*FZ_SYST_MAX;
    double fzp_syst_max = FzPmc*FZP_SYST_MAX;
    double k_syst_max = KMC*K_SYST_MAX;
    //
    // negative points
    for (int i=0; i<POINTS; ++i) {
      double x_ewk = 1.-ewk_syst_min + (ewk_syst_min)*i/POINTS;
      double x_fz  = fz_syst_min + (Fzmc-fz_syst_min)*i/POINTS;
      double x_fzp = fzp_syst_min + (FzPmc-fzp_syst_min)*i/POINTS;
      double x_k   = k_syst_min + (KMC-k_syst_min)*i/POINTS;
      double x_i   = i_syst_min + (Imc-i_syst_min)*i/POINTS;
      //
      double y_ewk= CalcABCD(Imc, Fzmc, FzPmc, KMC, x_ewk, Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
      double y_fz = CalcABCD(Imc, x_fz, FzPmc, KMC, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
      double y_fzp= CalcABCD(Imc, Fzmc, x_fzp, KMC, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
      double y_k  = CalcABCD(Imc, Fzmc, FzPmc, x_k, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
      double y_i  = CalcABCD(x_i, Fzmc, FzPmc, KMC, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
      //
      g_ewk.SetPoint(i,(x_ewk-1.)*100., 100.*fabs(y_ewk-SMC)/SMC);
      g_fz.SetPoint(i,(x_fz-Fzmc)*100./Fzmc, 100.*fabs(y_fz-SMC)/SMC);
      g_fzp.SetPoint(i,(x_fzp-FzPmc)*100./FzPmc, 100.*fabs(y_fzp-SMC)/SMC);
      g_i.SetPoint(i,(x_i-Imc)*100./Imc, 100.*fabs(y_i-SMC)/SMC);
      g_k.SetPoint(i,(x_k-KMC)*100./KMC, 100.*fabs(y_k-SMC)/SMC);
    }
    //
    // positive points
    for (int i=0; i<=POINTS; ++i) {
      double x_ewk = 1.+ewk_syst_max*i/POINTS; 
      double x_fz  = Fzmc+fz_syst_max*i/POINTS;
      double x_fzp = FzPmc+fzp_syst_max*i/POINTS;
      double x_k   = KMC+k_syst_max*i/POINTS;
      double x_i   = Imc+i_syst_max*i/POINTS;
      //
      double y_ewk= CalcABCD(Imc, Fzmc, FzPmc, KMC, x_ewk, Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
      double y_fz = CalcABCD(Imc, x_fz, FzPmc, KMC, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
      double y_fzp= CalcABCD(Imc, Fzmc, x_fzp, KMC, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
      double y_k  = CalcABCD(Imc, Fzmc, FzPmc, x_k, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
      double y_i  = CalcABCD(x_i, Fzmc, FzPmc, KMC, 1., Na, Nb, Nc, Nd, Ea,Eb,Ec,Ed);
      //
      g_ewk.SetPoint(i+POINTS,(x_ewk-1.)*100., 100.*fabs(y_ewk-SMC)/SMC);
      g_fz.SetPoint(i+POINTS,(x_fz-Fzmc)*100./Fzmc, 100.*fabs(y_fz-SMC)/SMC);
      g_fzp.SetPoint(i+POINTS,(x_fzp-FzPmc)*100./FzPmc, 100.*fabs(y_fzp-SMC)/SMC);
      g_i.SetPoint(i+POINTS,(x_i-Imc)*100./Imc, 100.*fabs(y_i-SMC)/SMC);
      g_k.SetPoint(i+POINTS,(x_k-KMC)*100./KMC, 100.*fabs(y_k-SMC)/SMC);
    }
    TString yaxis("(S-S_{mc})/S_{mc} (%)");
    TCanvas c;
    g_ewk.SetLineWidth(2);
    g_ewk.GetXaxis()->SetTitle("EWK Variation (%)");
    g_ewk.GetYaxis()->SetTitle(yaxis);
    g_ewk.Draw("AL");
    c.Print("ewk_syst_variation.C");
    //
    g_fz.SetLineWidth(2);
    g_fz.GetXaxis()->SetTitle("F_{z} Variation (%)");
    g_fz.GetYaxis()->SetTitle(yaxis);
    g_fz.Draw("AL");
    c.Print("fz_syst_variation.C");
    //
    g_fzp.SetLineWidth(2);
    g_fzp.GetXaxis()->SetTitle("F_{z}' Variation (%)");
    g_fzp.GetYaxis()->SetTitle(yaxis);
    g_fzp.Draw("AL");
    c.Print("fzp_syst_variation.C");
    //
    g_i.SetLineWidth(2);
    g_i.GetXaxis()->SetTitle("I Variation (%)");
    g_i.GetYaxis()->SetTitle(yaxis);
    g_i.Draw("AL");
    c.Print("i_syst_variation.C");
    //
    g_k.SetLineWidth(2);
    g_k.GetXaxis()->SetTitle("K Variation (%)");
    g_k.GetYaxis()->SetTitle(yaxis);
    g_k.Draw("AL");
    c.Print("k_syst_variation.C");
    //
    // ******************************************************************
    //
    //
    // 
    double err_ewk = std::max(fabs(SMC-S_EWK_PLUS),fabs(SMC-S_EWK_MINUS));
    double err_fz = fabs(SMC-S_FZ);
    double err_fzp = fabs(SMC-S_FZP);
    double err_i  = fabs(SMC-S_I);
    double err_k = fabs(SMC-S_K);
    //
    double DS_syst = sqrt(err_ewk*err_ewk + err_fz*err_fz + err_fzp*err_fzp+
			  err_i*err_i + err_k*err_k);
    //
    cout << "********************************************************" << endl;
    cout << "Signal Prediction: " << S << "+-" << DS << "(stat) +-"
	 << DS_syst << "(syst)"  << endl;
    cout << "stat error: " << 100.*DS/S <<"%" << endl;
    cout << "syt  error: " << 100.*DS_syst/S<< "%"  << endl;
    cout << "********************************************************" << endl;
    cout << "Parameters used in calculation: " << endl;
    cout << "I=  " << I << "+-" << dI << endl;
    cout << "Fz= " << Fz << "+-" << dFz << endl;
    cout << "FzP=" << FzP << "+-" << dFzP << endl;
    cout << "EWK error assumed to be: " << ewkerror << endl;
    cout << endl;
    cout << "ABCD Regions population:" << endl;
    cout << "A:  N=" << Na << ", sig=" << a_sig << ", qcd=" << a_qcd
         << ", ewk=" << a_ewk << endl;
    cout << "B:  N=" << Nb << ", sig=" << b_sig << ", qcd=" << b_qcd
         << ", ewk=" << b_ewk << endl;
    cout << "C:  N=" << Nc << ", sig=" << c_sig << ", qcd=" << c_qcd
         << ", ewk=" << c_ewk << endl;
    cout << "D:  N=" << Nd << ", sig=" << d_sig << ", qcd=" << d_qcd
         << ", ewk=" << d_ewk << endl;
    cout << endl;
    cout << "Parameters from MC: " << endl;
    cout << "I=  " << Imc << "+-" << dImc << endl;
    cout << "Fz= " << Fzmc << "+-" << dFzmc << endl;
    cout << "FzP=" << FzPmc << "+-" << dFzPmc << endl;
    cout << endl;
    cout << "Real value of K=" << KMC << endl;
    cout << "Real value of Signal=" << SMC << endl;
    cout << endl;
    cout << "Difference Measured - MC value (% wrt MC value except K=1): " 
	 << endl;
    cout << "Fz : " << dfz  << ", (" << dfz*100./Fzmc << "%)" << endl;
    cout << "FzP: " << dfzp << ", (" << dfzp*100./FzPmc << "%)"  << endl;
    cout << "I  : " << di   << ", (" << di*100./Imc << "%)"  << endl;
    cout << "K  : " << fk   << ", (" << fk*100./1. << "%)"  << endl;
    cout << endl;
    //
    cout << "DETAILS OF THE CALCULATION" << endl;
    cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
    cout << "Statistical Error Summary: " << endl;
    cout << "due to Fz = "<< SpFz*dFz<< ", ("<<SpFz*dFz*100./S << "%)"<< endl;
    cout << "due to FzP= "<< SpFzP*dFzP
	 << ", ("<<SpFzP*dFzP*100./S << "%)"<< endl; 
    cout << "due to  I = "<< SpI*dI
	 << ", ("<<SpI*dI*100./S << "%)"<< endl; 
    cout << "due to Na = "<< SpNa*sqrt(Na)
	 << ", ("<< SpNa*sqrt(Na)*100./S << "%)"<< endl; 
    cout << "due to Nb = "<< SpNb*sqrt(Nb)
	 << ", ("<< SpNb*sqrt(Nb)*100./S << "%)"<< endl; 
    cout << "due to Nc = "<< SpNc*sqrt(Nc)
	 << ", ("<< SpNc*sqrt(Nc)*100./S << "%)"<< endl; 
    cout << "due to Nd = "<< SpNd*sqrt(Nd)
	 << ", ("<< SpNd*sqrt(Nd)*100./S << "%)"<< endl; 
    cout << "Total Statistical Error: " 
	 << DS << ", (" << DS*100./S << "%)"<< endl;
    cout << "Stat Error percentages are wrt S prediction, not S mc" << endl;
    cout << endl;
    cout << "Systematic Error Summary:" << endl;
    cout << "due to k   = " << err_k << " ( " << err_k*100./S  << "%)" << endl;
    cout << "due to Fz  = " << err_fz << " ( " << err_fz*100./S  << "%)" << endl;
    cout << "due to FzP = " << err_fzp << " ( " << err_fzp*100./S  << "%)" << endl;
    cout << "due to I   = " << err_i << " ( " << err_i*100./S  << "%)" << endl;
    cout << "due to EWK = " << err_ewk << " ( " << err_ewk*100./S  << "%)" << endl;

    cout << "Syst Error percentages are wrt S prediction, not S mc" << endl;
  }
  //
  //
  if (mcOnly < 0 && mcOnly >-0.75) {  // select value -0.5 that gives the
                                      // string parser
    cout << "=======================================================" << endl;
    cout << "Calculating ABCD Result and Stat Error Assuming MC ONLY"  << endl;
    cout << "=======================================================" << endl;
    cout << "All input parameters that the user have inserted will be "
	 << "ignored and recalculated from MC" << endl;
    cout << "This option will not give you systematics estimation" << endl;
    // recalculate the basic quantities
    I = (a_sig + b_sig) / (a_sig + b_sig + c_sig + d_sig);
    dI = sqrt(I*(1-I)/(a_sig + b_sig + c_sig + d_sig));
    Fz = a_sig/b_sig;
    double e =a_sig/(a_sig + b_sig);
    double de = sqrt(e*(1-e)/(a_sig + b_sig));
    double alpha = de/(2.*Fz-e);
    dFz = alpha/(1-alpha);
    FzP = d_sig/c_sig;
    double ep =d_sig/(c_sig + d_sig);
    double dep = sqrt(ep*(1-ep)/(c_sig + d_sig));
    double alphap = dep/(2.*FzP-ep);
    dFzP = alphap/(1-alphap);
    //
    double KMC = (d_qcd/c_qcd)/(a_qcd/b_qcd);
    //
    // now everything is done from data + input
    double A = (1.0-I)*(FzP-Fz);
    double B = I*(FzP+1.0)*(FzP*(c_sig+c_qcd)-(d_sig+d_qcd)) + 
      (1+Fz)*(1-I)*((a_sig+a_qcd)-dFzP*(b_sig+b_qcd));
    double C = I*(1.+Fz)*(1.+FzP)*((d_sig+d_qcd)*(b_sig+b_qcd) - 
				   (a_sig+a_qcd)*(c_sig+c_qcd));
    //
    // signal calculation:
    double S = Trionym(A,B,C, a_sig+b_sig);

    // the errors now:
    // calculate the statistical error now:
    double  ApI=0, ApFz=0, ApFzP=0, ApNa=0, ApNb=0, ApNc=0, ApNd=0;
    double  BpI=0, BpFz=0, BpFzP=0, BpNa=0, BpNb=0, BpNc=0, BpNd=0;
    double  CpI=0, CpFz=0, CpFzP=0, CpNa=0, CpNb=0, CpNc=0, CpNd=0;
    double  SpI=0, SpFz=0, SpFzP=0, SpNa=0, SpNb=0, SpNc=0, SpNd=0;
    //
    double Na = a_sig+a_qcd+a_ewk, Nb = b_sig+b_qcd+b_ewk;
    double Nc=c_sig+c_qcd+c_ewk,   Nd = d_sig+d_qcd+d_ewk;
    double Ea = a_ewk, Eb = b_ewk, Ec=c_ewk, Ed = d_ewk;
    if (A != 0) {
      
      ApI   = -(FzP-Fz);
      ApFz  = -(1.0-I);
      ApFzP = (1.0-I);
      ApNa  = 0.0;
      ApNb  = 0.0;
      ApNc  = 0.0;
      ApNd  = 0.0;
      
      BpI   = (FzP+1.0)*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0+Fz)*((Na-Ea)-FzP*(Nb-Eb));
      BpFz  = I*(FzP+1.0)*(Nc-Ec)+(1.0-I)*((Na-Ea)-FzP*(Nb-Eb));
      BpFzP = I*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0-I)*(1.0+Fz)*(Nb-Eb);
      BpNa  =  (1.0-I)*(1.0+Fz);
      BpNb  = -(1.0-I)*(1.0+Fz)*FzP;
      BpNc  = I*(FzP+1.0)*Fz;
      BpNd  = -I*(FzP+1.0);
      
      CpI   = (1.0+Fz)*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec)); 
      CpFz  = I*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpFzP = I*(1.0+Fz)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpNa  = -I*(1.0+Fz)*(1.0+FzP)*(Nc-Ec);
      CpNb  =  I*(1.0+Fz)*(1.0+FzP)*(Nd-Ed);
      CpNc  = -I*(1.0+Fz)*(1.0+FzP)*(Na-Ea);
      CpNd  =  I*(1.0+Fz)*(1.0+FzP)*(Nb-Eb);
      
      SpI   = (-BpI   + (B*BpI   -2.0*ApI*C   -2.0*A*CpI)  /fabs(2.0*A*S+B)- 2.0*ApI*S)  /(2.0*A);
      SpFz  = (-BpFz  + (B*BpFz  -2.0*ApFz*C  -2.0*A*CpFz) /fabs(2.0*A*S+B)- 2.0*ApFz*S) /(2.0*A);
      SpFzP = (-BpFzP + (B*BpFzP -2.0*ApFzP*C -2.0*A*CpFzP)/fabs(2.0*A*S+B)- 2.0*ApFzP*S)/(2.0*A);
      SpNa  = (-BpNa  + (B*BpNa  -2.0*ApNa*C  -2.0*A*CpNa) /fabs(2.0*A*S+B)- 2.0*ApNa*S) /(2.0*A);
      SpNb  = (-BpNb  + (B*BpNb  -2.0*ApNb*C  -2.0*A*CpNb) /fabs(2.0*A*S+B)- 2.0*ApNb*S) /(2.0*A);
      SpNc  = (-BpNc  + (B*BpNc  -2.0*ApNc*C  -2.0*A*CpNc) /fabs(2.0*A*S+B)- 2.0*ApNc*S) /(2.0*A);
      SpNd  = (-BpNd  + (B*BpNd  -2.0*ApNd*C  -2.0*A*CpNd) /fabs(2.0*A*S+B)- 2.0*ApNd*S) /(2.0*A);
    }
    else {
      BpI   = (FzP+1.0)*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0+Fz)*((Na-Ea)-FzP*(Nb-Eb));
      BpFz  = I*(FzP+1.0)*(Nc-Ec)+(1.0-I)*((Na-Ea)-FzP*(Nb-Eb));
      BpFzP = I*(Fz*(Nc-Ec)-(Nd-Ed))-(1.0-I)*(1.0+Fz)*(Nb-Eb);
      BpNa  =  (1.0-I)*(1.0+Fz);
      BpNb  = -(1.0-I)*(1.0+Fz)*FzP;
      BpNc  = I*(FzP+1.0)*Fz;
      BpNd  = -I*(FzP+1.0);
      
      CpI   = (1.0+Fz)*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec)); 
      CpFz  = I*(1.0+FzP)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpFzP = I*(1.0+Fz)*((Nd-Ed)*(Nb-Eb)-(Na-Ea)*(Nc-Ec));
      CpNa  = -I*(1.0+Fz)*(1.0+FzP)*(Nc-Ec);
      CpNb  =  I*(1.0+Fz)*(1.0+FzP)*(Nd-Ed);
      CpNc  = -I*(1.0+Fz)*(1.0+FzP)*(Na-Ea);
      CpNd  =  I*(1.0+Fz)*(1.0+FzP)*(Nb-Eb);
      
      SpI   = -CpI/B+C*BpI/(B*B);
      SpFz  = -CpFz/B+C*BpFz/(B*B);
      SpFzP = -CpFzP/B+C*BpFzP/(B*B);
      SpNa  = -CpNa/B+C*BpNa/(B*B);
      SpNb  = -CpNb/B+C*BpNb/(B*B);
      SpNc  = -CpNc/B+C*BpNc/(B*B);
      SpNd  = -CpNd/B+C*BpNd/(B*B);
    }
    double  DS;
    DS = sqrt( SpI*dI*SpI*dI + SpFz*dFz*SpFz*dFz + SpFzP*dFzP*SpFzP*dFzP  +
	       SpNa*SpNa*Na + SpNb*SpNb*Nb + SpNc*SpNc*Nc + SpNd*SpNd*Nd );
    // warning: S here denotes the method prediction ..........
    cout << "********************************************************" << endl;
    cout << "Signal Prediction: " << S << "+-" << DS << "(stat)" << endl;
    cout << "********************************************************" << endl;
    cout << "Parameters used in calculation: " << endl;
    cout << "I=  " << I << "+-" << dI << endl;
    cout << "Fz= " << Fz << "+-" << dFz << endl;
    cout << "FzP=" << FzP << "+-" << dFzP << endl;
    cout << endl;
    cout << "ABCD Regions population:" << endl;
    cout << "A:  N=" << Na << ", sig=" << a_sig << ", qcd=" << a_qcd
	 << ", ewk=" << a_ewk << endl;
    cout << "B:  N=" << Nb << ", sig=" << b_sig << ", qcd=" << b_qcd
	 << ", ewk=" << b_ewk << endl;
    cout << "C:  N=" << Nc << ", sig=" << c_sig << ", qcd=" << c_qcd
	 << ", ewk=" << c_ewk << endl;
    cout << "D:  N=" << Nd << ", sig=" << d_sig << ", qcd=" << d_qcd
	 << ", ewk=" << d_ewk << endl;
    cout << "K value from MC: " << KMC << endl;
    cout << endl;
    cout << "Statistical Error Summary: " << endl;
    cout << "due to Fz = "<< SpFz*dFz<< ", ("<<SpFz*dFz*100./S << "%)"<< endl;
    cout << "due to FzP= "<< SpFzP*dFzP
	 << ", ("<<SpFzP*dFzP*100./S << "%)"<< endl; 
    cout << "due to  I = "<< SpI*dI
	 << ", ("<<SpI*dI*100./S << "%)"<< endl; 
    cout << "due to Na = "<< SpNa*sqrt(Na)
	 << ", ("<< SpNa*sqrt(Na)*100./S << "%)"<< endl; 
    cout << "due to Nb = "<< SpNb*sqrt(Nb)
	 << ", ("<< SpNb*sqrt(Nb)*100./S << "%)"<< endl; 
    cout << "due to Nc = "<< SpNc*sqrt(Nc)
	 << ", ("<< SpNc*sqrt(Nc)*100./S << "%)"<< endl; 
    cout << "due to Nd = "<< SpNd*sqrt(Nd)
	 << ", ("<< SpNd*sqrt(Nd)*100./S << "%)"<< endl; 
    cout << "Total Statistical Error: " 
	 << DS << ", (" << DS*100./S << "%)"<< endl;
    cout << "Stat Error percentages are wrt S prediction, not S mc" << endl;
  }


}

void plotMaker(TString histoName, TString wzsignal,
	       vector<TString> file, vector<TString> type, 
	       vector<double> weight, TString xtitle)
{
  gROOT->Reset();
  gROOT->ProcessLine(".L tdrstyle.C"); 
  gROOT->ProcessLine("setTDRStyle()");
  // automatic recognition of histogram dimension
  int NBins = 0; double min = 0; double max = -1;
  for (int i=0; i< (int) file.size(); ++i) {
    if (weight[i]>0) {
      TFile f(file[i]);
      TH1F *h = (TH1F*) f.Get(histoName);
      NBins = h->GetNbinsX();
      min = h->GetBinLowEdge(1);
      max = h->GetBinLowEdge(NBins+1);
      break;
    }
  }
  if (NBins ==0 || (max<min)) {
    std::cout << "PlotCombiner::abcd error: Could not find valid histograms in file"
              << std::endl;
    abort();
  }
  cout << "Histograms with "<< NBins <<" bins  and range " << min << "-" << max  << endl;
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


//
// reads the ABCD string and returns its value
// value is whatever it exists after the = sign and
// before the comma or the parenthesis
//
// if the string is not contained returns -1
// if there is no value, but the string is contained -0.5
// if there is an error in the algorithm return -99 and print error
// else returns its value
double searchABCDstring(TString abcdString, TString keyword)
{
  int size = keyword.Sizeof();
  int existsEntry = abcdString.Index(keyword);
  //
  if (existsEntry==-1) return -1.;
  //
  TString previousVal = abcdString(existsEntry-1);
  if (!(previousVal == "," || previousVal == " " || 
	previousVal == "(" )) return -1.;
  //
  TString afterVal = abcdString(existsEntry+size-1);
  //std::cout << "afterVal=" << afterVal << std::endl;
  if (afterVal =="," || afterVal==")") return -0.5;
  else if (afterVal != "=") return -1.;
  //
  // now find the comma or the parenthesis after the = sign
  int comma = abcdString.Index(",",existsEntry);
  //std::cout << "first comma=" << comma << std::endl;
  if (comma<0) comma = abcdString.Index(")",existsEntry);
  if (comma<0) {
    std::cout << "Error in parcing abcd line, chech syntax " 
	      << std::endl;
    return -99.;
  }
  TString svalue=abcdString(existsEntry+size,comma-existsEntry-size);
  std::cout << "Found ABCD parameter "<< keyword
	    << " with value "  << svalue << endl;
  // convert this to a float
  double value = svalue.Atof();
  return value;

}


double Trionym(double a, double b, double c, double sum)
{
  if (a==0) {
    return -c/b;
  }
  double D2 = b*b - 4.*a*c;
  //return (-b + sqrt(D2)) / (2.*a);
  if (D2 > 0) {
    double s1 = (-b + sqrt(D2)) / (2.*a);
    double s2 = (-b - sqrt(D2)) / (2.*a);
    double solution =   fabs(s1-sum)<fabs(s2-sum)?s1:s2;
    return solution;
  }
  else  {
    return -1.;  
  }
}

//
// the naming of the variables and the order is in this way for historical
// reasons
// your complains for different Nd_ and nd to G.Daskalakis :P
//
double CalcABCD
(double I, double Fz, double FzP, double K, double ewk,
 double Na_, double Nb_, double Nc_, double Nd_ ,
 double Ea_, double Eb_, double Ec_, double Ed_)
{
  double A, B, C;
  A = (1.0-I)*(FzP-K*Fz);
  B = I*(FzP+1.0)*(K*Fz*(Nc_-ewk*Ec_)-(Nd_-ewk*Ed_))+
    (1.0-I)*(1.0+Fz)*(K*(Na_-ewk*Ea_)-FzP*(Nb_-ewk*Eb_));
  C = I*(1.0+Fz)*(1.0+FzP)*((Nd_-ewk*Ed_)*(Nb_-ewk*Eb_)-
			    K*(Na_-ewk*Ea_)*(Nc_-ewk*Ec_));
  //
  return Trionym(A, B, C, Na_+Nb_-Ea_-Eb_);

}
