#include "TObject.h"
#include "SeedPtFunction.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include <string>
#include <TString.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TFrame.h>
#include <TH2.h>
#include <TH1.h>
#include <TH1D.h>
#include <TF1.h>
#include <TFormula.h>
#include <TGraphErrors.h>
#include <TGraph.h>
#include <TMath.h>


class SeedPtScale : public TObject {

private:

   TString dname;
   TString Dir;
   TString det_id;
   TString suffixps ;
   TString pname ;

   int rbin;
   double xbsz ;
   double ybsz ;

   // debug tools
   bool debug;
   TCanvas* cv;
   char dbplot[11];
   TString plot_id ;
   
   TString plot01;
   TString plot02;
   TString plot03;
   TString plot04;
   TString plot05;
   TString plot06;

   TCanvas* c1;
   TCanvas* c2;
   TCanvas* c3;
   TCanvas* c4;
   TCanvas* c5;
   TCanvas* c6;


public:

   SeedPtScale();

   ~SeedPtScale();


   void PtScale( int type, int st, double h1, double h2, int idx, int np );
   void deBugger( TF1* fitfunc, TString funcName ,TH1D* histo, double L2, double H2, int color );
   bool BadFitting( TF1* fitfunc, TH1D* histo );
  
   ClassDef(SeedPtScale, 1);

};

#if !defined(__CINT__)
    ClassImp(SeedPtScale);
#endif

