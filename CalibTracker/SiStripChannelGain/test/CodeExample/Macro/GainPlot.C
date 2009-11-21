

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

int Color [] = {2,4,1,8,6,7,3,9,5};
int Marker[] = {21,22,23,20,29,3,2};
int Style [] = {1,2,5,7,9,10};

#include<vector>
#include<tdrstyle.C>

TObject* GetObjectFromPath(TDirectory* File, const char* Path);
void SaveCanvas(TCanvas* c, char* path, bool OnlyPPNG=false);
void DrawStatBox(TObject** Histos, std::vector<char*> legend, bool Mean               , double X=0.15, double Y=0.93, double W=0.15, double H=0.03);
void DrawLegend (TObject** Histos, std::vector<char*> legend, char* Title, char* Style, double X=0.80, double Y=0.93, double W=0.20, double H=0.05);
void DrawSuperposedHistos(TH1D** Histos, std::vector<char*> legend, char* Style,  char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax);
void DrawTH2D   (TH2D**    Histos, std::vector<char*> legend, char* Style, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax);

TF1*  getLandau(TH1* InputHisto, double* FitResults, double LowRange=50, double HighRange=5400);
TH1D* ChargeToMPV(TH2* InputHisto, char* Name, bool DivideByX);

void GainPlot()
{
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


   unsigned int  tree_Index;
   unsigned int  tree_DetId;
   unsigned char tree_APVId;
   unsigned char tree_SubDet;
   float         tree_x;
   float         tree_y;
   float         tree_z;
   float         tree_Eta;
   float         tree_R;
   float         tree_Phi;
   float         tree_Thickness;
   float         tree_FitMPV;
   float         tree_FitMPVErr;
   float         tree_FitWidth;
   float         tree_FitWidthErr;
   float         tree_FitChi2NDF;
   double        tree_Gain;
   double        tree_PrevGain;
   double        tree_NEntries;

   TFile* f1     = new TFile("file:../Gains_Tree.root");
   TTree *t1     = (TTree*)GetObjectFromPath(f1,"SiStripCalib/APVGain");

   t1->SetBranchAddress("Index"             ,&tree_Index      );
   t1->SetBranchAddress("DetId"             ,&tree_DetId      );
   t1->SetBranchAddress("APVId"             ,&tree_APVId      );
   t1->SetBranchAddress("SubDet"            ,&tree_SubDet     );
   t1->SetBranchAddress("x"                 ,&tree_x          );
   t1->SetBranchAddress("y"                 ,&tree_y          );
   t1->SetBranchAddress("z"                 ,&tree_z          );
   t1->SetBranchAddress("Eta"               ,&tree_Eta        );
   t1->SetBranchAddress("R"                 ,&tree_R          );
   t1->SetBranchAddress("Phi"               ,&tree_Phi        );
   t1->SetBranchAddress("Thickness"         ,&tree_Thickness  );
   t1->SetBranchAddress("FitMPV"            ,&tree_FitMPV     );
   t1->SetBranchAddress("FitMPVErr"         ,&tree_FitMPVErr  );
   t1->SetBranchAddress("FitWidth"          ,&tree_FitWidth   );
   t1->SetBranchAddress("FitWidthErr"       ,&tree_FitWidthErr);
   t1->SetBranchAddress("FitChi2NDF"        ,&tree_FitChi2NDF );
   t1->SetBranchAddress("Gain"              ,&tree_Gain       );
   t1->SetBranchAddress("PrevGain"          ,&tree_PrevGain   );
   t1->SetBranchAddress("NEntries"          ,&tree_NEntries   );


   TH2D* ChargeDistrib  = (TH2D*)GetObjectFromPath(f1,"SiStripCalib/Charge_Vs_Index");
   TH2D* ChargeDistribA = (TH2D*)GetObjectFromPath(f1,"SiStripCalib/Charge_Vs_Index_Absolute");

   TH2D* Charge_Vs_PathlengthTIB   = (TH2D*)GetObjectFromPath(f1,"SiStripCalib/Charge_Vs_PathlengthTIB");
   TH2D* Charge_Vs_PathlengthTOB   = (TH2D*)GetObjectFromPath(f1,"SiStripCalib/Charge_Vs_PathlengthTOB");
   TH2D* Charge_Vs_PathlengthTIDP  = (TH2D*)GetObjectFromPath(f1,"SiStripCalib/Charge_Vs_PathlengthTIDP");
   TH2D* Charge_Vs_PathlengthTIDM  = (TH2D*)GetObjectFromPath(f1,"SiStripCalib/Charge_Vs_PathlengthTIDM");
   TH2D* Charge_Vs_PathlengthTID   = (TH2D*)Charge_Vs_PathlengthTIDP->Clone("Charge_Vs_PathlengthTID");
         Charge_Vs_PathlengthTID      ->Add(Charge_Vs_PathlengthTIDM);
   TH2D* Charge_Vs_PathlengthTECP1 = (TH2D*)GetObjectFromPath(f1,"SiStripCalib/Charge_Vs_PathlengthTECP1");
   TH2D* Charge_Vs_PathlengthTECP2 = (TH2D*)GetObjectFromPath(f1,"SiStripCalib/Charge_Vs_PathlengthTECP2");
   TH2D* Charge_Vs_PathlengthTECM1 = (TH2D*)GetObjectFromPath(f1,"SiStripCalib/Charge_Vs_PathlengthTECM1");
   TH2D* Charge_Vs_PathlengthTECM2 = (TH2D*)GetObjectFromPath(f1,"SiStripCalib/Charge_Vs_PathlengthTECM2");
   TH2D* Charge_Vs_PathlengthTECP  = (TH2D*)Charge_Vs_PathlengthTECP1->Clone("Charge_Vs_PathlengthTECP");
         Charge_Vs_PathlengthTECP     ->Add(Charge_Vs_PathlengthTECP2);
   TH2D* Charge_Vs_PathlengthTECM  = (TH2D*)Charge_Vs_PathlengthTECM1->Clone("Charge_Vs_PathlengthTECM");
         Charge_Vs_PathlengthTECM     ->Add(Charge_Vs_PathlengthTECM2);
   TH2D* Charge_Vs_PathlengthTEC1  = (TH2D*)Charge_Vs_PathlengthTECP1->Clone("Charge_Vs_PathlengthTEC1");
         Charge_Vs_PathlengthTEC1     ->Add(Charge_Vs_PathlengthTECM1);
   TH2D* Charge_Vs_PathlengthTEC2  = (TH2D*)Charge_Vs_PathlengthTECP2->Clone("Charge_Vs_PathlengthTEC2");
         Charge_Vs_PathlengthTEC2     ->Add(Charge_Vs_PathlengthTECM2); 
   TH2D* Charge_Vs_PathlengthTEC   = (TH2D*)Charge_Vs_PathlengthTECP ->Clone("Charge_Vs_PathlengthTEC");
         Charge_Vs_PathlengthTEC      ->Add(Charge_Vs_PathlengthTECM );

   TH2D* Charge_Vs_PathlengthThin  = (TH2D*)Charge_Vs_PathlengthTEC1->Clone("Charge_Vs_PathlengthThin");
         Charge_Vs_PathlengthThin     ->Add(Charge_Vs_PathlengthTIB );
         Charge_Vs_PathlengthThin     ->Add(Charge_Vs_PathlengthTID );
   TH2D* Charge_Vs_PathlengthThick = (TH2D*)Charge_Vs_PathlengthTEC2->Clone("Charge_Vs_PathlengthThin");
         Charge_Vs_PathlengthThick    ->Add(Charge_Vs_PathlengthTOB );



   TH1D* MPV_Vs_PathlengthTIB      = ChargeToMPV(Charge_Vs_PathlengthTIB  ,"MPV_Vs_PathlengthTIB"  , true);
   TH1D* MPV_Vs_PathlengthTID      = ChargeToMPV(Charge_Vs_PathlengthTID  ,"MPV_Vs_PathlengthTID"  , true);
// TH1D* MPV_Vs_PathlengthTIDP     = ChargeToMPV(Charge_Vs_PathlengthTIDP ,"MPV_Vs_PathlengthTIDP" , true);
// TH1D* MPV_Vs_PathlengthTIDM     = ChargeToMPV(Charge_Vs_PathlengthTIDM ,"MPV_Vs_PathlengthTIDM" , true);
   TH1D* MPV_Vs_PathlengthTOB      = ChargeToMPV(Charge_Vs_PathlengthTOB  ,"MPV_Vs_PathlengthTOB"  , true);
// TH1D* MPV_Vs_PathlengthTEC      = ChargeToMPV(Charge_Vs_PathlengthTEC  ,"MPV_Vs_PathlengthTEC"  , true);
// TH1D* MPV_Vs_PathlengthTECP     = ChargeToMPV(Charge_Vs_PathlengthTECP ,"MPV_Vs_PathlengthTECP" , true);
// TH1D* MPV_Vs_PathlengthTECM     = ChargeToMPV(Charge_Vs_PathlengthTECM ,"MPV_Vs_PathlengthTECM" , true);
   TH1D* MPV_Vs_PathlengthTEC1     = ChargeToMPV(Charge_Vs_PathlengthTEC1 ,"MPV_Vs_PathlengthTEC1" , true);
   TH1D* MPV_Vs_PathlengthTEC2     = ChargeToMPV(Charge_Vs_PathlengthTEC2 ,"MPV_Vs_PathlengthTEC2" , true);
// TH1D* MPV_Vs_PathlengthTECP1    = ChargeToMPV(Charge_Vs_PathlengthTECP1,"MPV_Vs_PathlengthTECP1", true);
// TH1D* MPV_Vs_PathlengthTECP2    = ChargeToMPV(Charge_Vs_PathlengthTECP2,"MPV_Vs_PathlengthTECP2", true);
// TH1D* MPV_Vs_PathlengthTECM1    = ChargeToMPV(Charge_Vs_PathlengthTECM1,"MPV_Vs_PathlengthTECM1", true);
// TH1D* MPV_Vs_PathlengthTECM2    = ChargeToMPV(Charge_Vs_PathlengthTECM2,"MPV_Vs_PathlengthTECM2", true);
   TH1D* MPV_Vs_PathlengthThin     = ChargeToMPV(Charge_Vs_PathlengthThin ,"MPV_Vs_PathlengthThin" , true);
   TH1D* MPV_Vs_PathlengthThick    = ChargeToMPV(Charge_Vs_PathlengthThick,"MPV_Vs_PathlengthThick", true);

   TH2D* MPV_Vs_EtaTIB  = new TH2D("MPV_Vs_EtaTIB" ,"MPV_Vs_EtaTIB" , 50, -3.0, 3.0, 300, 0, 600);
   TH2D* MPV_Vs_EtaTID  = new TH2D("MPV_Vs_EtaTID" ,"MPV_Vs_EtaTID" , 50, -3.0, 3.0, 300, 0, 600);
   TH2D* MPV_Vs_EtaTOB  = new TH2D("MPV_Vs_EtaTOB" ,"MPV_Vs_EtaTOB" , 50, -3.0, 3.0, 300, 0, 600);
   TH2D* MPV_Vs_EtaTEC  = new TH2D("MPV_Vs_EtaTEC" ,"MPV_Vs_EtaTEC" , 50, -3.0, 3.0, 300, 0, 600);
   TH2D* MPV_Vs_EtaTEC1 = new TH2D("MPV_Vs_EtaTEC1","MPV_Vs_EtaTEC1", 50, -3.0, 3.0, 300, 0, 600);
   TH2D* MPV_Vs_EtaTEC2 = new TH2D("MPV_Vs_EtaTEC2","MPV_Vs_EtaTEC2", 50, -3.0, 3.0, 300, 0, 600);

   TH2D* MPV_Vs_PhiTIB  = new TH2D("MPV_Vs_PhiTIB" ,"MPV_Vs_PhiTIB" , 50, -3.4, 3.4, 300, 0, 600);
   TH2D* MPV_Vs_PhiTID  = new TH2D("MPV_Vs_PhiTID" ,"MPV_Vs_PhiTID" , 50, -3.4, 3.4, 300, 0, 600);
   TH2D* MPV_Vs_PhiTOB  = new TH2D("MPV_Vs_PhiTOB" ,"MPV_Vs_PhiTOB" , 50, -3.4, 3.4, 300, 0, 600);
   TH2D* MPV_Vs_PhiTEC  = new TH2D("MPV_Vs_PhiTEC" ,"MPV_Vs_PhiTEC" , 50, -3.4, 3.4, 300, 0, 600);
   TH2D* MPV_Vs_PhiTEC1 = new TH2D("MPV_Vs_PhiTEC1","MPV_Vs_PhiTEC1", 50, -3.4, 3.4, 300, 0, 600);
   TH2D* MPV_Vs_PhiTEC2 = new TH2D("MPV_Vs_PhiTEC2","MPV_Vs_PhiTEC2", 50, -3.4, 3.4, 300, 0, 600);

   TH2D* NoMPV          = new TH2D("NoMPV"         ,"NoMPV"         ,350, -350, 350, 240, 0, 120);

   TH1D* MPVs           = new TH1D("MPVs"          ,"MPVs"          ,                300, 0, 600);
   TH1D* MPVs320        = new TH1D("MPVs320"       ,"MPVs320"       ,                300, 0, 600);
   TH1D* MPVs500        = new TH1D("MPVs500"       ,"MPVs500"       ,                300, 0, 600);

   TH1D* MPVError       = new TH1D("MPVError"      ,"MPVError"      ,                150, 0, 150);
   TH2D* MPVErrorVsMPV  = new TH2D("MPVErrorVsMPV" ,"MPVErrorVsMPV" ,300,    0, 600, 150, 0, 150);
   TH2D* MPVErrorVsEta  = new TH2D("MPVErrorVsEta" ,"MPVErrorVsEta" , 50, -3.0, 3.0, 150, 0, 150); 
   TH2D* MPVErrorVsPhi  = new TH2D("MPVErrorVsPhi" ,"MPVErrorVsPhi" , 50, -3.4, 3.4, 150, 0, 150);             
   TH2D* MPVErrorVsN    = new TH2D("MPVErrorVsN"   ,"MPVErrorVsN"   ,500,    0,1000, 150, 0, 150);              




   TH1D* ChargeTIB      = new TH1D("ChargeTIB"     ,"ChargeTIB"     ,               1000, 0,2000);
   TH1D* ChargeTID      = new TH1D("ChargeTID"     ,"ChargeTID"     ,               1000, 0,2000);
   TH1D* ChargeTIDP     = new TH1D("ChargeTIDP"    ,"ChargeTIDP"    ,               1000, 0,2000);
   TH1D* ChargeTIDM     = new TH1D("ChargeTIDM"    ,"ChargeTIDM"    ,               1000, 0,2000);
   TH1D* ChargeTOB      = new TH1D("ChargeTOB"     ,"ChargeTOB"     ,               1000, 0,2000);
   TH1D* ChargeTEC      = new TH1D("ChargeTEC"     ,"ChargeTEC"     ,               1000, 0,2000);
   TH1D* ChargeTECP     = new TH1D("ChargeTECP"    ,"ChargeTECP"    ,               1000, 0,2000);
   TH1D* ChargeTECM     = new TH1D("ChargeTECM"    ,"ChargeTECM"    ,               1000, 0,2000);
   TH1D* ChargeTEC1     = new TH1D("ChargeTEC1"    ,"ChargeTEC1"    ,               1000, 0,2000);
   TH1D* ChargeTEC2     = new TH1D("ChargeTEC2"    ,"ChargeTEC2"    ,               1000, 0,2000);
   TH1D* ChargeTECP1    = new TH1D("ChargeTECP1"   ,"ChargeTECP1"   ,               1000, 0,2000);
   TH1D* ChargeTECP2    = new TH1D("ChargeTECP2"   ,"ChargeTECP2"   ,               1000, 0,2000);
   TH1D* ChargeTECM1    = new TH1D("ChargeTECM1"   ,"ChargeTECM1"   ,               1000, 0,2000);
   TH1D* ChargeTECM2    = new TH1D("ChargeTECM2"   ,"ChargeTECM2"   ,               1000, 0,2000);

   TH1D* ChargeAbsTIB   = new TH1D("ChargeAbsTIB"  ,"ChargeAbsTIB"  ,                500, 0,2000);
   TH1D* ChargeAbsTID   = new TH1D("ChargeAbsTID"  ,"ChargeAbsTID"  ,                500, 0,2000);
   TH1D* ChargeAbsTIDP  = new TH1D("ChargeAbsTIDP" ,"ChargeAbsTIDP" ,                500, 0,2000);
   TH1D* ChargeAbsTIDM  = new TH1D("ChargeAbsTIDM" ,"ChargeAbsTIDM" ,                500, 0,2000);
   TH1D* ChargeAbsTOB   = new TH1D("ChargeAbsTOB"  ,"ChargeAbsTOB"  ,                500, 0,2000);
   TH1D* ChargeAbsTEC   = new TH1D("ChargeAbsTEC"  ,"ChargeAbsTEC"  ,                500, 0,2000);
   TH1D* ChargeAbsTECP  = new TH1D("ChargeAbsTECP" ,"ChargeAbsTECP" ,                500, 0,2000);
   TH1D* ChargeAbsTECM  = new TH1D("ChargeAbsTECM" ,"ChargeAbsTECM" ,                500, 0,2000);
   TH1D* ChargeAbsTEC1  = new TH1D("ChargeAbsTEC1" ,"ChargeAbsTEC1" ,                500, 0,2000);
   TH1D* ChargeAbsTEC2  = new TH1D("ChargeAbsTEC2" ,"ChargeAbsTEC2" ,                500, 0,2000);
   TH1D* ChargeAbsTECP1 = new TH1D("ChargeAbsTECP1","ChargeAbsTECP1",                500, 0,2000);
   TH1D* ChargeAbsTECP2 = new TH1D("ChargeAbsTECP2","ChargeAbsTECP2",                500, 0,2000);
   TH1D* ChargeAbsTECM1 = new TH1D("ChargeAbsTECM1","ChargeAbsTECM1",                500, 0,2000);
   TH1D* ChargeAbsTECM2 = new TH1D("ChargeAbsTECM2","ChargeAbsTECM2",                500, 0,2000);



   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on the Tree          :");
   int TreeStep = t1->GetEntries()/50;if(TreeStep==0)TreeStep=1;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      TH1D* Proj         = ChargeDistrib ->ProjectionY("proj" ,tree_Index, tree_Index);
      TH1D* ProjAbsolute = ChargeDistribA->ProjectionY("projA",tree_Index, tree_Index);


      if(tree_SubDet==3                       ) MPV_Vs_EtaTIB ->Fill(tree_Eta,tree_FitMPV);
      if(tree_SubDet==4                       ) MPV_Vs_EtaTID ->Fill(tree_Eta,tree_FitMPV);
      if(tree_SubDet==5                       ) MPV_Vs_EtaTOB ->Fill(tree_Eta,tree_FitMPV);
      if(tree_SubDet==6                       ) MPV_Vs_EtaTEC ->Fill(tree_Eta,tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness<0.04) MPV_Vs_EtaTEC1->Fill(tree_Eta,tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness>0.04) MPV_Vs_EtaTEC2->Fill(tree_Eta,tree_FitMPV);

      if(tree_SubDet==3                       ) MPV_Vs_PhiTIB ->Fill(tree_Phi,tree_FitMPV);
      if(tree_SubDet==4                       ) MPV_Vs_PhiTID ->Fill(tree_Phi,tree_FitMPV);
      if(tree_SubDet==5                       ) MPV_Vs_PhiTOB ->Fill(tree_Phi,tree_FitMPV);
      if(tree_SubDet==6                       ) MPV_Vs_PhiTEC ->Fill(tree_Phi,tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness<0.04) MPV_Vs_PhiTEC1->Fill(tree_Phi,tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness>0.04) MPV_Vs_PhiTEC2->Fill(tree_Phi,tree_FitMPV);

                                                MPVs          ->Fill(         tree_FitMPV);
      if(                  tree_Thickness<0.04) MPVs320       ->Fill(         tree_FitMPV);
      if(                  tree_Thickness>0.04) MPVs500       ->Fill(         tree_FitMPV);


      if(tree_FitMPV<0                        ) NoMPV         ->Fill(tree_z ,tree_R);

                                                MPVError      ->Fill(         tree_FitMPVErr);    
                                                MPVErrorVsMPV ->Fill(tree_FitMPV,tree_FitMPVErr);
                                                MPVErrorVsEta ->Fill(tree_Eta,tree_FitMPVErr);
                                                MPVErrorVsPhi ->Fill(tree_Phi,tree_FitMPVErr);
                                                MPVErrorVsN   ->Fill(tree_NEntries,tree_FitMPVErr);


      if(tree_SubDet==3                       ) ChargeTIB  ->Add(Proj,1);
      if(tree_SubDet==4                       ) ChargeTID  ->Add(Proj,1);
      if(tree_SubDet==4 && tree_Eta<0         ) ChargeTIDM ->Add(Proj,1);
      if(tree_SubDet==4 && tree_Eta>0         ) ChargeTIDP ->Add(Proj,1);
      if(tree_SubDet==5                       ) ChargeTOB  ->Add(Proj,1);
      if(tree_SubDet==6                       ) ChargeTEC  ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Thickness<0.04) ChargeTEC1 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Thickness>0.04) ChargeTEC2 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta<0         ) ChargeTECP ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta>0         ) ChargeTECM ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness<0.04) ChargeTECM1 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness>0.04) ChargeTECM2 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness<0.04) ChargeTECP1 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness>0.04) ChargeTECP2 ->Add(Proj,1);


      if(tree_SubDet==3                       ) ChargeAbsTIB  ->Add(ProjAbsolute,1);
      if(tree_SubDet==4                       ) ChargeAbsTID  ->Add(ProjAbsolute,1);
      if(tree_SubDet==4 && tree_Eta<0         ) ChargeAbsTIDM ->Add(ProjAbsolute,1);
      if(tree_SubDet==4 && tree_Eta>0         ) ChargeAbsTIDP ->Add(ProjAbsolute,1);
      if(tree_SubDet==5                       ) ChargeAbsTOB  ->Add(ProjAbsolute,1);
      if(tree_SubDet==6                       ) ChargeAbsTEC  ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Thickness<0.04) ChargeAbsTEC1 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Thickness>0.04) ChargeAbsTEC2 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta<0         ) ChargeAbsTECP ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta>0         ) ChargeAbsTECM ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness<0.04) ChargeAbsTECM1 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness>0.04) ChargeAbsTECM2 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness<0.04) ChargeAbsTECP1 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness>0.04) ChargeAbsTECP2 ->Add(ProjAbsolute,1);



      delete Proj;
      delete ProjAbsolute;
   }printf("\n");

   TCanvas* c1;
   TObject** Histos = new TObject*[10];                
   std::vector<char*> legend;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = MPV_Vs_EtaTEC;                     legend.push_back("TEC");
   Histos[1] = MPV_Vs_EtaTIB;                     legend.push_back("TIB");
   Histos[2] = MPV_Vs_EtaTID;                     legend.push_back("TID");
   Histos[3] = MPV_Vs_EtaTOB;                     legend.push_back("TOB");
   DrawTH2D((TH2D**)Histos,legend, "", "module #eta", "MPV (ADC/mm)", -3.0,3.0, 0,500);
   DrawLegend (Histos,legend,"","P");
   DrawStatBox(Histos,legend,false);
   SaveCanvas(c1,"Pictures/MPV_Vs_EtaSubDet");
   delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPV_Vs_PhiTEC;                     legend.push_back("TEC");
    Histos[1] = MPV_Vs_PhiTIB;                     legend.push_back("TIB");
    Histos[2] = MPV_Vs_PhiTID;                     legend.push_back("TID");
    Histos[3] = MPV_Vs_PhiTOB;                     legend.push_back("TOB");
    DrawTH2D((TH2D**)Histos,legend, "", "module #phi", "MPV (ADC/mm)", -3.4,3.4, 0,500);
    DrawLegend(Histos,legend,"","P");
    DrawStatBox(Histos,legend,false);
    SaveCanvas(c1,"Pictures/MPV_Vs_PhiSubDet");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = NoMPV;                             legend.push_back("NoMPV");
    DrawTH2D((TH2D**)Histos,legend, "", "z (cm)", "R (cms)", 0,0, 0,0);
    SaveCanvas(c1,"Pictures/NoMPV", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    ChargeDistrib->GetXaxis()->SetNdivisions(5+500);
    Histos[0] = ChargeDistrib;                     legend.push_back("Charge Vs Index");
    //DrawTH2D((TH2D**)Histos,legend, "COLZ", "APV Index", "Charge (ADC/mm)", 0,0, 0,0);
    //c1->SetLogz(true);
    //SaveCanvas(c1,"Pictures/Charge", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVs320;                           legend.push_back("320 #mum");
    Histos[1] = MPVs500;                           legend.push_back("500 #mum");
    Histos[2] = MPVs;                              legend.push_back("320 + 500 #mum");
    DrawSuperposedHistos((TH1D**)Histos, legend, "",  "MPV (ADC/mm)", "#APVs", 0,500, 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true);
    SaveCanvas(c1,"Pictures/MPVs");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVError;                          legend.push_back("MPV Error");
    DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Error on MPV (ADC/mm)", "#APVs", 0,500, 0,0);
    DrawStatBox(Histos,legend,true);
    c1->SetLogy(true);
    SaveCanvas(c1,"Pictures/Error");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsMPV;                     legend.push_back("Error Vs MPV");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "MPV (ADC/mm)", "Error on MPV (ADC/mm)", 0,0, 0,0);
    c1->SetLogz(true);
    SaveCanvas(c1,"Pictures/Error_Vs_MPV", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsEta;                     legend.push_back("Error Vs Eta");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "module #eta", "Error on MPV (ADC/mm)", 0,0, 0,0);
    c1->SetLogz(true);
    SaveCanvas(c1,"Pictures/Error_Vs_Eta", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsPhi;                     legend.push_back("Error Vs Phi");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "module #phi", "Error on MPV (ADC/mm)", 0,0, 0,0);
    c1->SetLogz(true);
    SaveCanvas(c1,"Pictures/Error_Vs_Phi", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsN;                       legend.push_back("Error Vs Entries");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "Number of Entries", "Error on MPV (ADC/mm)", 0,0, 0,0);
    c1->SetLogz(true);
    SaveCanvas(c1,"Pictures/Error_Vs_N", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTEC;                         legend.push_back("TEC");
    Histos[1] = ChargeTIB;                         legend.push_back("TIB");
    Histos[2] = ChargeTID;                         legend.push_back("TID");
    Histos[3] = ChargeTOB;                         legend.push_back("TOB");
    DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Charge (ADC/mm)", "#clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,"Pictures/Charge");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTECP;                        legend.push_back("TEC+");
    Histos[1] = ChargeTECM;                        legend.push_back("TEC-");
    DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Charge (ADC/mm)", "#clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,"Pictures/ChargeTECSide");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTEC1;                        legend.push_back("TEC Thin");
    Histos[1] = ChargeTEC2;                        legend.push_back("TEC Thick");
    DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Charge (ADC/mm)", "#clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,"Pictures/ChargeTECThickness");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTIDP;                        legend.push_back("TID+");
    Histos[1] = ChargeTIDM;                        legend.push_back("TID-");
    DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Charge (ADC/mm)", "#clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,"Pictures/ChargeTIDSide");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTEC;                      legend.push_back("TEC");
    Histos[1] = ChargeAbsTIB;                      legend.push_back("TIB");
    Histos[2] = ChargeAbsTID;                      legend.push_back("TID");
    Histos[3] = ChargeAbsTOB;                      legend.push_back("TOB");
    DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Charge (ADC)", "#clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,"Pictures/ChargeAbs");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTECP;                     legend.push_back("TEC+");
    Histos[1] = ChargeAbsTECM;                     legend.push_back("TEC-");
    DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Charge (ADC)", "#clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,"Pictures/ChargeAbsTECSide");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTEC1;                     legend.push_back("TEC Thin");
    Histos[1] = ChargeAbsTEC2;                     legend.push_back("TEC Thick");
    DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Charge (ADC)", "#clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,"Pictures/ChargeAbsTECThickness");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTIDP;                     legend.push_back("TID+");
    Histos[1] = ChargeAbsTIDM;                     legend.push_back("TID-");
    DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Charge (ADC)", "#clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,"Pictures/ChargeAbsTIDSide");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPV_Vs_PathlengthThin;             legend.push_back("320 #mum");
    Histos[1] = MPV_Vs_PathlengthThick;            legend.push_back("500 #mum");
    DrawSuperposedHistos((TH1D**)Histos, legend, "HIST",  "pathlength (mm)", "MPV (ADC/mm)", 0,0 , 230,380);
    DrawLegend(Histos,legend,"","L");
    SaveCanvas(c1,"Pictures/MPV_Vs_Path");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPV_Vs_PathlengthTIB;              legend.push_back("TIB (320 #mum)");
    Histos[1] = MPV_Vs_PathlengthTID;              legend.push_back("TID (320 #mum)");
    Histos[2] = MPV_Vs_PathlengthTOB;              legend.push_back("TOB (500 #mum)");
    Histos[3] = MPV_Vs_PathlengthTEC1;             legend.push_back("TEC (320 #mum)");
    Histos[4] = MPV_Vs_PathlengthTEC2;             legend.push_back("TEC (500 #mum)");
    DrawSuperposedHistos((TH1D**)Histos, legend, "HIST",  "pathlength (mm)", "MPV (ADC/mm)", 0,0 , 230,380);
    DrawLegend(Histos,legend,"","L");
    SaveCanvas(c1,"Pictures/MPV_Vs_PathSubDet");
    delete c1;
}



TObject* GetObjectFromPath(TDirectory* File, const char* Path)
{
   string str(Path);
   size_t pos = str.find("/");

   if(pos < 256){
      string firstPart = str.substr(0,pos);
      string endPart   = str.substr(pos+1,str.length());
      TDirectory* TMP = (TDirectory*)File->Get(firstPart.c_str());
      if(TMP!=NULL)return GetObjectFromPath(TMP,endPart.c_str());

      printf("BUG\n");
      return NULL;
   }else{
      return File->Get(Path);
   }

}

void SaveCanvas(TCanvas* c, char* path, bool OnlyPPNG){
   char buff[1024];
   sprintf(buff,"%s.png",path);  c->SaveAs(buff);   if(OnlyPPNG)return;
   sprintf(buff,"%s.eps",path);  c->SaveAs(buff);
   sprintf(buff,"%s.c"  ,path);  c->SaveAs(buff);
}

void DrawLegend(TObject** Histos, std::vector<char*> legend, char* Title, char* Style, double X, double Y, double W, double H)
{
   int    N             = legend.size();

   if(strcmp(legend[0],"")!=0){
      TLegend* leg;
      leg = new TLegend(X,Y,X-W,Y - N*H);
      leg->SetFillColor(0);
      leg->SetBorderSize(0);
      if(strcmp(Title,"")!=0)leg->SetHeader(Title);

      for(int i=0;i<N;i++){
         TH2D* temp = (TH2D*)Histos[i]->Clone();
         temp->SetMarkerSize(1.3);
         leg->AddEntry(temp, legend[i] ,Style);
      }
      leg->Draw();
   }
}


void DrawStatBox(TObject** Histos, std::vector<char*> legend, bool Mean, double X, double Y, double W, double H)
{
   int    N             = legend.size();
   char   buffer[255];

   if(Mean)H*=3;
   for(int i=0;i<N;i++){
           TPaveText* stat = new TPaveText(X,Y-(i*H), X+W, Y-(i+1)*H, "NDC");
	   TH1* Histo = (TH1*)Histos[i];
           sprintf(buffer,"Entries : %i\n",(int)Histo->GetEntries());
           stat->AddText(buffer);

           if(Mean){
           sprintf(buffer,"Mean    : %6.2f\n",Histo->GetMean());
           stat->AddText(buffer);

           sprintf(buffer,"RMS     : %6.2f\n",Histo->GetRMS());
           stat->AddText(buffer);
           }

           stat->SetFillColor(0);
           stat->SetLineColor(Color[i]);
           stat->SetTextColor(Color[i]);
           stat->SetBorderSize(0);
           stat->SetMargin(0.05);
           stat->SetTextAlign(12);
           stat->Draw();
   }
}



void DrawTH2D(TH2D** Histos, std::vector<char*> legend, char* Style, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax)
{
   int    N             = legend.size();

   for(int i=0;i<N;i++){
        if(!Histos[i])continue;
        Histos[i]->SetTitle("");
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend);
        Histos[i]->GetYaxis()->SetTitle(Ylegend);
        Histos[i]->GetYaxis()->SetTitleOffset(1.60);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(0.3);
   }

   char Buffer[256];
   Histos[0]->Draw(Style);
   for(int i=1;i<N;i++){
        sprintf(Buffer,"%s same",Style);
        Histos[i]->Draw(Buffer);
   }
}

void DrawSuperposedHistos(TH1D** Histos, std::vector<char*> legend, char* Style,  char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax)
{
   int    N             = legend.size();

   double HistoMax      = -1;
   int    HistoHeighest = -1;

   for(int i=0;i<N;i++){
        if(!Histos[i])continue;
        Histos[i]->SetTitle("");
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend);
        Histos[i]->GetYaxis()->SetTitle(Ylegend);
        Histos[i]->GetYaxis()->SetTitleOffset(1.60);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetFillColor(0);
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(0.5);
        Histos[i]->SetLineColor(Color[i]);
        Histos[i]->SetLineWidth(2);

        if(Histos[i]->GetMaximum() >= HistoMax){
	   HistoMax      = Histos[i]->GetMaximum();
           HistoHeighest = i;
	}

   }

   char Buffer[256];
   Histos[HistoHeighest]->Draw(Style);
   for(int i=0;i<N;i++){        
        if(strcmp(Style,"")!=0){
           sprintf(Buffer,"same %s",Style);
        }else{
           sprintf(Buffer,"same");
        }
        Histos[i]->Draw(Buffer);
   }
}


TF1* getLandau(TH1* InputHisto, double* FitResults, double LowRange, double HighRange)
{
   FitResults[0]         = -0.5;  //MPV
   FitResults[1]         =  0;    //MPV error
   FitResults[2]         = -0.5;  //Width
   FitResults[3]         =  0;    //Width error
   FitResults[4]         = -0.5;  //Fit Chi2/NDF

   // perform fit with standard landau
   TF1* MyLandau = new TF1("MyLandau","landau",LowRange, HighRange);
   MyLandau->SetParameter(1,300);
   InputHisto->Fit("MyLandau","0QR WW");

   // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
   FitResults[0]         = MyLandau->GetParameter(1);  //MPV
   FitResults[1]         = MyLandau->GetParError(1);   //MPV error
   FitResults[2]         = MyLandau->GetParameter(2);  //Width
   FitResults[3]         = MyLandau->GetParError(2);   //Width error
   FitResults[4]         = MyLandau->GetChisquare() / MyLandau->GetNDF();  //Fit Chi2/NDF

   return MyLandau;
}

TH1D* ChargeToMPV(TH2* InputHisto, char* Name,  bool DivideByX)
{
   TH1D* toReturn = new TH1D(Name,Name,InputHisto->GetXaxis()->GetNbins(), InputHisto->GetXaxis()->GetXmin(), InputHisto->GetXaxis()->GetXmax() );
   double Results[4];

   for(int i=0;i<=InputHisto->GetXaxis()->GetNbins();i++){
      TH1D* proj   = InputHisto->ProjectionY("",i,i);
      if(proj->GetEntries()<50){delete proj;continue;}

      TF1*  landau = getLandau(proj,Results);

      if(DivideByX){
         toReturn->SetBinContent(i,Results[0] / InputHisto->GetXaxis()->GetBinCenter(i) );
         toReturn->SetBinError  (i,Results[1] / InputHisto->GetXaxis()->GetBinCenter(i) );
      }else{
         toReturn->SetBinContent(i,Results[0]);
         toReturn->SetBinError  (i,Results[1]);
      }
      delete landau;
      delete proj;
   }
   return toReturn;
}

