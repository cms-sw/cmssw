

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

   
   system("mkdir -p Pictures/Charge");
   PlotMacro_Core("file:../../Data_Run_247252_to_247990_PCL/Gains_Tree.root", "file:../../Data_Run_247992_to_247992_PCL/Gains_Tree.root", "SiStripCalib"          , "Checks");
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


   TFile* f2     = new TFile(input2.c_str());
   TTree *t2     = (TTree*)GetObjectFromPath(f2,moduleName+"/APVGain");
   TH2D* ChargeDistrib2  = (TH2D*)GetObjectFromPath(f2,moduleName+"/Charge_Vs_Index");
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


       if(tree2_FitMPV>0 and tree2_FitMPV<150){
          TH1D* Proj1         = ChargeDistrib1 ->ProjectionY("proj1" ,tree1_Index, tree1_Index, "e");
          TH1D* Proj2         = ChargeDistrib2 ->ProjectionY("proj2" ,tree2_Index, tree2_Index, "e");

          TCanvas* c1 = new TCanvas("c1", "c1", 600,600);
//          c1->SetLogy(true);


          TH1D* frame = new TH1D("frame", "frame", 1,0, 1000);
          frame->GetXaxis()->SetNdivisions(505);
          frame->SetTitle("");
          frame->SetStats(kFALSE);
          frame->GetXaxis()->SetTitle("charge/path-length (ADC/mm)");
          frame->GetYaxis()->SetTitle("#clusters");
          frame->SetMaximum(std::max(tree1_NEntries, tree2_NEntries)*0.04);
          frame->SetMinimum(0.0);
          frame->GetYaxis()->SetTitleOffset(1.50);
          frame->Draw();

          //Proj1->Scale(1.0/Proj1->Integral());
          Proj1->SetLineColor(2);
          Proj1->SetLineWidth(2);          
          Proj1->Draw("H same"); 

          TF1* Fit1 = new TF1("MyLandau1","landau", 0, 2000);
          Fit1->SetParameters(tree1_NEntries/tree1_FitWidth, tree1_FitMPV, tree1_FitWidth);
          Fit1->SetLineColor(2);
          Fit1->SetLineWidth(2);
          Fit1->Draw("L same");

          TF1* Fit1b = getPeakOfLandau(Proj1, "MyLandau1bis");
          Fit1b->SetLineColor(8);
          Fit1b->SetLineWidth(2);
          Fit1b->Draw("L same");

          printf("period1 MPV vs new MPV --> %6.2f  vs %6.2f\n", tree1_FitMPV, Fit1b->GetParameter(1));


          //Proj2->Scale(1.0/Proj2->Integral());
          Proj2->SetLineColor(4);
          Proj2->SetLineWidth(2);
          Proj2->Draw("H same");

          TF1* Fit2 = new TF1("MyLandau2","landau", 0, 2000);
          Fit2->SetParameters(tree2_NEntries/tree2_FitWidth, tree2_FitMPV, tree2_FitWidth);
          Fit2->SetLineColor(4);
          Fit2->SetLineWidth(2);
          Fit2->Draw("L same");


          TF1* Fit2b = getPeakOfLandau(Proj2, "MyLandau2bis");
          Fit2b->SetLineColor(8);
          Fit2b->SetLineWidth(2);
          Fit2b->Draw("L same");

          printf("period2 MPV vs new MPV --> %6.2f  vs %6.2f\n", tree2_FitMPV, Fit2b->GetParameter(1));

          char buffer[256];
          sprintf(buffer, "Pictures/Charge/SubDet%i_Id%i_Apv%i_N%i.png",tree1_SubDet, tree1_DetId, tree1_APVId, (int)tree1_NEntries);
          c1->SaveAs(buffer);
       }




      if(tree1_APVId==0 && PreviousId>0){
         double Mean = (MPV2/NAPV) / (MPV1/NAPV);
         if(Mean<Min)Min=Mean;
         if(Mean>Max)Max=Mean;
         if(NAPV>0) fprintf(pFile,"%i  %f\n",PreviousId,Mean);
         NAPV=0; MPV1=0; MPV2=0;
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
