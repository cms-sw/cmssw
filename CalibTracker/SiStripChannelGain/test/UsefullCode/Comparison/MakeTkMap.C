

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TPaveText.h"
#include "PlotFunction.h"


#include<vector>
#include<tdrstyle.C>

void PlotMacro_Core(string input, string input2, string moduleName, string output);

int DataType = 2;

void MakeTkMap(){
   gROOT->Reset();
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.05);
   gStyle->SetPadBottomMargin(0.10);
   gStyle->SetPadRightMargin (0.18);
   gStyle->SetPadLeftMargin  (0.13);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetCanvasColor(0);
   gStyle->SetBarOffset(0);

   
   system("mkdir -p Pictures/Charge");
//   PlotMacro_Core("file:../../Data_Run_247252_to_247990_PCL/Gains_Tree.root", "file:../../Data_Run_247992_to_247992_PCL/Gains_Tree.root", "SiStripCalib"          , "Checks");
     PlotMacro_Core("file:../../Data_Run_247992_to_247992_PCL/Gains_Tree.root", "file:../../Data_Run_247992_to_247992_PCL/Gains_Tree.root", "SiStripCalib"          , "Checks");

}


TF1* getPeakOfLandau(TH1* InputHisto, char* name, double LowRange=50, double HighRange=5400)
{ 
   // perform fit with standard landau
   TF1* MyLandau = new TF1(name,"landau",LowRange, HighRange);
   MyLandau->SetParameter(1,300);
   InputHisto->Fit(MyLandau,"0QR WW");
   return MyLandau;
}


void PlotMacro_Core(string input, string input2, string moduleName, string output)
{
   FILE* pFile;
   TCanvas* c1;
   TObject** Histos = new TObject*[10];                
   std::vector<string> legend;

   unsigned int  tree1_Index;
   unsigned int  tree1_DetId;
   unsigned char tree1_APVId;
   unsigned char tree1_SubDet;
   float         tree1_x;
   float         tree1_y;
   float         tree1_z;
   float         tree1_Eta;
   float         tree1_R;
   float         tree1_Phi;
   float         tree1_Thickness;
   float         tree1_FitMPV;
   float         tree1_FitMPVErr;
   float         tree1_FitWidth;
   float         tree1_FitWidthErr;
   float         tree1_FitChi2NDF;
   double        tree1_Gain;
   double        tree1_PrevGain;
   double        tree1_NEntries;
   bool          tree1_isMasked;


   TFile* f1     = new TFile(input.c_str());
   TTree *t1     = (TTree*)GetObjectFromPath(f1,moduleName+"/APVGain");
   TH2D* ChargeDistrib1  = (TH2D*)GetObjectFromPath(f1,moduleName+"/Charge_Vs_Index");
   t1->SetBranchAddress("Index"             ,&tree1_Index      );
   t1->SetBranchAddress("DetId"             ,&tree1_DetId      );
   t1->SetBranchAddress("APVId"             ,&tree1_APVId      );
   t1->SetBranchAddress("SubDet"            ,&tree1_SubDet     );
   t1->SetBranchAddress("x"                 ,&tree1_x          );
   t1->SetBranchAddress("y"                 ,&tree1_y          );
   t1->SetBranchAddress("z"                 ,&tree1_z          );
   t1->SetBranchAddress("Eta"               ,&tree1_Eta        );
   t1->SetBranchAddress("R"                 ,&tree1_R          );
   t1->SetBranchAddress("Phi"               ,&tree1_Phi        );
   t1->SetBranchAddress("Thickness"         ,&tree1_Thickness  );
   t1->SetBranchAddress("FitMPV"            ,&tree1_FitMPV     );
   t1->SetBranchAddress("FitMPVErr"         ,&tree1_FitMPVErr  );
   t1->SetBranchAddress("FitWidth"          ,&tree1_FitWidth   );
   t1->SetBranchAddress("FitWidthErr"       ,&tree1_FitWidthErr);
   t1->SetBranchAddress("FitChi2NDF"        ,&tree1_FitChi2NDF );
   t1->SetBranchAddress("Gain"              ,&tree1_Gain       );
   t1->SetBranchAddress("PrevGain"          ,&tree1_PrevGain   );
   t1->SetBranchAddress("NEntries"          ,&tree1_NEntries   );
   t1->SetBranchAddress("isMasked"          ,&tree1_isMasked   );

   pFile = fopen("TkMap.txt","w");

   unsigned int PreviousId = 0;
   unsigned int NAPV       = 0;
   double       MPV1       = 0;
   double Min=9999, Max=-9999, Mean=0;
 
   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on the Tree          :");
   int TreeStep = t1->GetEntries()/50;if(TreeStep==0)TreeStep=1;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);

      if(tree1_APVId==0 && PreviousId>0){
         double Mean = MPV1/NAPV;
         if(Mean<Min)Min=Mean;
         if(Mean>Max)Max=Mean;
         if(NAPV>0) fprintf(pFile,"%i  %f\n",PreviousId,Mean);
         NAPV=0; MPV1=0;
      }
      PreviousId = tree1_DetId;
      if(tree1_FitMPV<=0)continue;
      NAPV++;
      MPV1+=tree1_FitMPV;
   }printf("\n");
   fclose(pFile);

   printf("Min=%f - Max=%f\n",Min,Max);
}
