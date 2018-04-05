

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

void MakeChargeDistribution(){
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

   
   system("mkdir Pictures");
   PlotMacro_Core("file:../7TeVData_170249_to_170901/Gains_Tree.root", "file:../7TeVData/Gains_Tree.root", "SiStripCalib"          , "Checks");
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


   unsigned int  tree2_Index;
   unsigned int  tree2_DetId;
   unsigned char tree2_APVId;
   unsigned char tree2_SubDet;
   float         tree2_x;
   float         tree2_y;
   float         tree2_z;
   float         tree2_Eta;
   float         tree2_R;
   float         tree2_Phi;
   float         tree2_Thickness;
   float         tree2_FitMPV;
   float         tree2_FitMPVErr;
   float         tree2_FitWidth;
   float         tree2_FitWidthErr;
   float         tree2_FitChi2NDF;
   double        tree2_Gain;
   double        tree2_PrevGain;
   double        tree2_NEntries;
   bool          tree2_isMasked;


   TFile* f1     = new TFile(input.c_str());
   TTree *t1     = (TTree*)GetObjectFromPath(f1,moduleName+"/APVGain");

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


   TFile* f2     = new TFile(input2.c_str());
   TTree *t2     = (TTree*)GetObjectFromPath(f2,moduleName+"/APVGain");

   t2->SetBranchAddress("Index"             ,&tree2_Index      );
   t2->SetBranchAddress("DetId"             ,&tree2_DetId      );
   t2->SetBranchAddress("APVId"             ,&tree2_APVId      ); 
   t2->SetBranchAddress("SubDet"            ,&tree2_SubDet     );
   t2->SetBranchAddress("x"                 ,&tree2_x          ); 
   t2->SetBranchAddress("y"                 ,&tree2_y          );
   t2->SetBranchAddress("z"                 ,&tree2_z          );
   t2->SetBranchAddress("Eta"               ,&tree2_Eta        ); 
   t2->SetBranchAddress("R"                 ,&tree2_R          ); 
   t2->SetBranchAddress("Phi"               ,&tree2_Phi        );
   t2->SetBranchAddress("Thickness"         ,&tree2_Thickness  ); 
   t2->SetBranchAddress("FitMPV"            ,&tree2_FitMPV     ); 
   t2->SetBranchAddress("FitMPVErr"         ,&tree2_FitMPVErr  ); 
   t2->SetBranchAddress("FitWidth"          ,&tree2_FitWidth   );
   t2->SetBranchAddress("FitWidthErr"       ,&tree2_FitWidthErr);
   t2->SetBranchAddress("FitChi2NDF"        ,&tree2_FitChi2NDF ); 
   t2->SetBranchAddress("Gain"              ,&tree2_Gain       );
   t2->SetBranchAddress("PrevGain"          ,&tree2_PrevGain   );
   t2->SetBranchAddress("NEntries"          ,&tree2_NEntries   );
   t2->SetBranchAddress("isMasked"          ,&tree2_isMasked   ); 

   pFile = fopen("GainDiff.txt","w");

   unsigned int PreviousId = 0;
   unsigned int NAPV       = 0;
   double       MPV1       = 0;
   double       MPV2       = 0;

   double Min = 10000;
   double Max =-10000;

   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on the Tree          :");
   int TreeStep = t1->GetEntries()/50;if(TreeStep==0)TreeStep=1;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      t2->GetEntry(ientry);

      if(tree1_APVId==0 && PreviousId>0){
         double Mean = (MPV2-MPV1)/NAPV;
         if(Mean<Min)Min=Mean;
         if(Mean>Max)Max=Mean;
         if(NAPV>0) fprintf(pFile,"%i  %f\n",PreviousId,Mean);
         NAPV=0; MPV1=0; MPV2=0;
         exit(0);
      }
      PreviousId = tree1_DetId;
      if(tree1_FitMPV<=0 || tree2_FitMPV<=0)continue;
      NAPV++;
      MPV1+=tree1_FitMPV;
      MPV2+=tree2_FitMPV;
   }printf("\n");
   fclose(pFile);

   printf("Min=%f - Max=%f\n",Min,Max);

}
