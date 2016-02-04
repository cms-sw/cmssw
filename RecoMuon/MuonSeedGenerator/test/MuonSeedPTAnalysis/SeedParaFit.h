#include "TObject.h"
#include "SeedPtFunction.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include <string>
#include <TString.h>
#include <TSystem.h>
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

class SeedParaFit : public TObject {

private:

   TString dname ;
   TString fName ;
   TString suffixps ;
   float nsigmas ;
   double bsz ;
   int rbin ;
   bool debug ;
   bool debugAll ;

   TFile *file ;
   FILE* dfile;
   FILE* logfile;

   char det[6];
   char dphi_case[20];
   TString dphi_type ;
   TString det_id ;

   TString plot01;
   TString plot02;
   TString plot03;
   TString plot04;
   TString plot05;

   TCanvas* c1;
   TCanvas* c2;
   TCanvas* c3;
   TCanvas* c4;
   TCanvas* c5;

   TH2F* heta_dphiPt;

   // debug tools
   TCanvas* cv;
   char dbplot[11];
   TString plot_id ;

   SeedPtFunction* ptFunc ;

public:

   SeedParaFit();

   ~SeedParaFit();


   void ParaFit( int type, int st, double h1, double h2, int np );
   
   vector<int> preFitter( TF1* ffunc, TString falias, double h1, double h2, int asize, Double_t* xa, Double_t* ya );

   void PrintTitle();
   void PrintEnd();

   ClassDef(SeedParaFit, 1);

};

#if !defined(__CINT__)
    ClassImp(SeedParaFit);
#endif

