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
#include "tdrstyle.C"

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"

using namespace fwlite;
using namespace std;
using namespace edm;
#endif




void PlotMacro_Core(string input, string moduleName, string output, string TextToPrint);
TF1*  getLandau(TH1* InputHisto, double* FitResults, double LowRange=50, double HighRange=5400);
TH1D* ChargeToMPV(TH2* InputHisto, string Name, bool DivideByX);



void PlotMacro(string TextToPrint_="CMS Preliminary 2015"){
   system("mkdir Pictures");
   PlotMacro_Core("file:Gains_Tree.root"     , "SiStripCalib/"          , "Pictures/Gains"     , TextToPrint_ + "  -  Particle Gain");
   PlotMacro_Core("file:Validation_Tree.root", "SiStripCalibValidation/", "Pictures/Validation", TextToPrint_ + "  -  Gain Validation");
}


void PlotMacro_Core(string input, string moduleName, string output, string TextToPrint)
{
   FILE* pFile;
   TCanvas* c1;
   TObject** Histos = new TObject*[10];                
   std::vector<string> legend;

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
   float         tree_FitNorm;
   double        tree_Gain;
   double        tree_PrevGain;
   double        tree_PrevGainTick;
   double        tree_NEntries;
   bool          tree_isMasked;

   TFile* f1     = new TFile(input.c_str());
   TTree *t1     = (TTree*)GetObjectFromPath(f1,moduleName+"APVGain");

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
   t1->SetBranchAddress("FitNorm"           ,&tree_FitNorm    );
   t1->SetBranchAddress("Gain"              ,&tree_Gain       );
   t1->SetBranchAddress("PrevGain"          ,&tree_PrevGain   );
   t1->SetBranchAddress("PrevGainTick"      ,&tree_PrevGainTick);
   t1->SetBranchAddress("NEntries"          ,&tree_NEntries   );
   t1->SetBranchAddress("isMasked"          ,&tree_isMasked   );


   TH2D* ChargeDistrib  = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_Index");
   TH2D* ChargeDistribA = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_Index_Absolute");

   TH2D* Charge_Vs_PathlengthTIB   = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTIB");
   TH2D* Charge_Vs_PathlengthTOB   = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTOB");
   TH2D* Charge_Vs_PathlengthTIDP  = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTIDP");
   TH2D* Charge_Vs_PathlengthTIDM  = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTIDM");
   TH2D* Charge_Vs_PathlengthTID   = (TH2D*)Charge_Vs_PathlengthTIDP->Clone("Charge_Vs_PathlengthTID");
         Charge_Vs_PathlengthTID      ->Add(Charge_Vs_PathlengthTIDM);
   TH2D* Charge_Vs_PathlengthTECP1 = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTECP1");
   TH2D* Charge_Vs_PathlengthTECP2 = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTECP2");
   TH2D* Charge_Vs_PathlengthTECM1 = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTECM1");
   TH2D* Charge_Vs_PathlengthTECM2 = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTECM2");
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



   TH1D* Gains          = new TH1D("Gains"         ,"Gains"         ,                300, 0, 2);
   TH1D* MPVs           = new TH1D("MPVs"          ,"MPVs"          ,                300, 0, 600);
   TH1D* MPVs320        = new TH1D("MPVs320"       ,"MPVs320"       ,                300, 0, 600);
   TH1D* MPVs500        = new TH1D("MPVs500"       ,"MPVs500"       ,                300, 0, 600);
   TH1D* MPVsTIB        = new TH1D("MPVsTIB"       ,"MPVsTIB"       ,                300, 0, 600);
   TH1D* MPVsTID        = new TH1D("MPVsTID"       ,"MPVsTID"       ,                300, 0, 600);
   TH1D* MPVsTIDP       = new TH1D("MPVsTIDP"      ,"MPVsTIDP"      ,                300, 0, 600);
   TH1D* MPVsTIDM       = new TH1D("MPVsTIDM"      ,"MPVsTIDM"      ,                300, 0, 600);
   TH1D* MPVsTOB        = new TH1D("MPVsTOB"       ,"MPVsTOB"       ,                300, 0, 600);
   TH1D* MPVsTEC        = new TH1D("MPVsTEC"       ,"MPVsTEC"       ,                300, 0, 600);
   TH1D* MPVsTECP       = new TH1D("MPVsTECP"      ,"MPVsTECP"      ,                300, 0, 600);
   TH1D* MPVsTECM       = new TH1D("MPVsTECM"      ,"MPVsTECM"      ,                300, 0, 600);
   TH1D* MPVsTEC1       = new TH1D("MPVsTEC1"      ,"MPVsTEC1"      ,                300, 0, 600);
   TH1D* MPVsTEC2       = new TH1D("MPVsTEC2"      ,"MPVsTEC2"      ,                300, 0, 600);
   TH1D* MPVsTECP1      = new TH1D("MPVsTECP1"     ,"MPVsTECP1"     ,                300, 0, 600);
   TH1D* MPVsTECP2      = new TH1D("MPVsTECP2"     ,"MPVsTECP2"     ,                300, 0, 600);
   TH1D* MPVsTECM1      = new TH1D("MPVsTECM1"     ,"MPVsTECM1"     ,                300, 0, 600);
   TH1D* MPVsTECM2      = new TH1D("MPVsTECM2"     ,"MPVsTECM2"     ,                300, 0, 600);


   TH1D* MPVError       = new TH1D("MPVError"      ,"MPVError"      ,                150, 0, 150);
   TH2D* MPVErrorVsMPV  = new TH2D("MPVErrorVsMPV" ,"MPVErrorVsMPV" ,300,    0, 600, 150, 0, 150);
   TH2D* MPVErrorVsEta  = new TH2D("MPVErrorVsEta" ,"MPVErrorVsEta" , 50, -3.0, 3.0, 150, 0, 150); 
   TH2D* MPVErrorVsPhi  = new TH2D("MPVErrorVsPhi" ,"MPVErrorVsPhi" , 50, -3.4, 3.4, 150, 0, 150);             
   TH2D* MPVErrorVsN    = new TH2D("MPVErrorVsN"   ,"MPVErrorVsN"   ,500,    0,1000, 150, 0, 150);              



   TH1D* ChargePIB      = new TH1D("ChargePIB"     ,"ChargePIB"     ,               2000, 0,4000);
   TH1D* ChargePIE      = new TH1D("ChargePIE"     ,"ChargePIE"     ,               2000, 0,4000);
   TH1D* ChargeTIB      = new TH1D("ChargeTIB"     ,"ChargeTIB"     ,               2000, 0,4000);
   TH1D* ChargeTID      = new TH1D("ChargeTID"     ,"ChargeTID"     ,               2000, 0,4000);
   TH1D* ChargeTIDP     = new TH1D("ChargeTIDP"    ,"ChargeTIDP"    ,               2000, 0,4000);
   TH1D* ChargeTIDM     = new TH1D("ChargeTIDM"    ,"ChargeTIDM"    ,               2000, 0,4000);
   TH1D* ChargeTOB      = new TH1D("ChargeTOB"     ,"ChargeTOB"     ,               2000, 0,4000);
   TH1D* ChargeTEC      = new TH1D("ChargeTEC"     ,"ChargeTEC"     ,               2000, 0,4000);
   TH1D* ChargeTECP     = new TH1D("ChargeTECP"    ,"ChargeTECP"    ,               2000, 0,4000);
   TH1D* ChargeTECM     = new TH1D("ChargeTECM"    ,"ChargeTECM"    ,               2000, 0,4000);
   TH1D* ChargeTEC1     = new TH1D("ChargeTEC1"    ,"ChargeTEC1"    ,               2000, 0,4000);
   TH1D* ChargeTEC2     = new TH1D("ChargeTEC2"    ,"ChargeTEC2"    ,               2000, 0,4000);
   TH1D* ChargeTECP1    = new TH1D("ChargeTECP1"   ,"ChargeTECP1"   ,               2000, 0,4000);
   TH1D* ChargeTECP2    = new TH1D("ChargeTECP2"   ,"ChargeTECP2"   ,               2000, 0,4000);
   TH1D* ChargeTECM1    = new TH1D("ChargeTECM1"   ,"ChargeTECM1"   ,               2000, 0,4000);
   TH1D* ChargeTECM2    = new TH1D("ChargeTECM2"   ,"ChargeTECM2"   ,               2000, 0,4000);

   TH1D* ChargeAbsPIB   = new TH1D("ChargeAbsPIB"  ,"ChargeAbsPIB"  ,               1000, 0,4000);
   TH1D* ChargeAbsPIE   = new TH1D("ChargeAbsPIE"  ,"ChargeAbsPIE"  ,               1000, 0,4000);
   TH1D* ChargeAbsTIB   = new TH1D("ChargeAbsTIB"  ,"ChargeAbsTIB"  ,               1000, 0,4000);
   TH1D* ChargeAbsTID   = new TH1D("ChargeAbsTID"  ,"ChargeAbsTID"  ,               1000, 0,4000);
   TH1D* ChargeAbsTIDP  = new TH1D("ChargeAbsTIDP" ,"ChargeAbsTIDP" ,               1000, 0,4000);
   TH1D* ChargeAbsTIDM  = new TH1D("ChargeAbsTIDM" ,"ChargeAbsTIDM" ,               1000, 0,4000);
   TH1D* ChargeAbsTOB   = new TH1D("ChargeAbsTOB"  ,"ChargeAbsTOB"  ,               1000, 0,4000);
   TH1D* ChargeAbsTEC   = new TH1D("ChargeAbsTEC"  ,"ChargeAbsTEC"  ,               1000, 0,4000);
   TH1D* ChargeAbsTECP  = new TH1D("ChargeAbsTECP" ,"ChargeAbsTECP" ,               1000, 0,4000);
   TH1D* ChargeAbsTECM  = new TH1D("ChargeAbsTECM" ,"ChargeAbsTECM" ,               1000, 0,4000);
   TH1D* ChargeAbsTEC1  = new TH1D("ChargeAbsTEC1" ,"ChargeAbsTEC1" ,               1000, 0,4000);
   TH1D* ChargeAbsTEC2  = new TH1D("ChargeAbsTEC2" ,"ChargeAbsTEC2" ,               1000, 0,4000);
   TH1D* ChargeAbsTECP1 = new TH1D("ChargeAbsTECP1","ChargeAbsTECP1",               1000, 0,4000);
   TH1D* ChargeAbsTECP2 = new TH1D("ChargeAbsTECP2","ChargeAbsTECP2",               1000, 0,4000);
   TH1D* ChargeAbsTECM1 = new TH1D("ChargeAbsTECM1","ChargeAbsTECM1",               1000, 0,4000);
   TH1D* ChargeAbsTECM2 = new TH1D("ChargeAbsTECM2","ChargeAbsTECM2",               1000, 0,4000);

   TH1D* DiffWRTPrevGainPIB      = new TH1D("DiffWRTPrevGainPIB"     ,"DiffWRTPrevGainPIB"     ,               250, 0,2);
   TH1D* DiffWRTPrevGainPIE      = new TH1D("DiffWRTPrevGainPIE"     ,"DiffWRTPrevGainPIE"     ,               250, 0,2);
   TH1D* DiffWRTPrevGainTIB      = new TH1D("DiffWRTPrevGainTIB"     ,"DiffWRTPrevGainTIB"     ,               250, 0,2);
   TH1D* DiffWRTPrevGainTID      = new TH1D("DiffWRTPrevGainTID"     ,"DiffWRTPrevGainTID"     ,               250, 0,2);
   TH1D* DiffWRTPrevGainTOB      = new TH1D("DiffWRTPrevGainTOB"     ,"DiffWRTPrevGainTOB"     ,               250, 0,2);
   TH1D* DiffWRTPrevGainTEC      = new TH1D("DiffWRTPrevGainTEC"     ,"DiffWRTPrevGainTEC"     ,               250, 0,2);

   TH2D* GainVsPrevGainPIB      = new TH2D("GainVsPrevGainPIB"     ,"GainVsPrevGainPIB"     ,               100, 0,2, 100, 0,2);
   TH2D* GainVsPrevGainPIE      = new TH2D("GainVsPrevGainPIE"     ,"GainVsPrevGainPIE"     ,               100, 0,2, 100, 0,2);
   TH2D* GainVsPrevGainTIB      = new TH2D("GainVsPrevGainTIB"     ,"GainVsPrevGainTIB"     ,               100, 0,2, 100, 0,2);
   TH2D* GainVsPrevGainTID      = new TH2D("GainVsPrevGainTID"     ,"GainVsPrevGainTID"     ,               100, 0,2, 100, 0,2);
   TH2D* GainVsPrevGainTOB      = new TH2D("GainVsPrevGainTOB"     ,"GainVsPrevGainTOB"     ,               100, 0,2, 100, 0,2);
   TH2D* GainVsPrevGainTEC      = new TH2D("GainVsPrevGainTEC"     ,"GainVsPrevGainTEC"     ,               100, 0,2, 100, 0,2);

   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on the Tree          :");
   int TreeStep = t1->GetEntries()/50;if(TreeStep==0)TreeStep=1;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);

      int bin = ChargeDistrib->GetXaxis()->FindBin(tree_Index);
      TH1D* Proj         = ChargeDistrib ->ProjectionY("proj" ,bin, bin);
      TH1D* ProjAbsolute = ChargeDistribA->ProjectionY("projA",bin, bin);

      if(tree_SubDet>=3 && tree_FitMPV<0      ) NoMPV         ->Fill(tree_z ,tree_R);
      if(tree_SubDet>=3 && tree_FitMPV>=0){

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

      if(tree_FitMPV>0                        ) Gains         ->Fill(         tree_Gain  );
                                                MPVs          ->Fill(         tree_FitMPV);
      if(                  tree_Thickness<0.04) MPVs320       ->Fill(         tree_FitMPV);
      if(                  tree_Thickness>0.04) MPVs500       ->Fill(         tree_FitMPV);
      if(tree_SubDet==3                       ) MPVsTIB       ->Fill(         tree_FitMPV);
      if(tree_SubDet==4                       ) MPVsTID       ->Fill(         tree_FitMPV);
      if(tree_SubDet==4 && tree_Eta<0         ) MPVsTIDM      ->Fill(         tree_FitMPV);
      if(tree_SubDet==4 && tree_Eta>0         ) MPVsTIDP      ->Fill(         tree_FitMPV);
      if(tree_SubDet==5                       ) MPVsTOB       ->Fill(         tree_FitMPV);
      if(tree_SubDet==6                       ) MPVsTEC       ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness<0.04) MPVsTEC1      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness>0.04) MPVsTEC2      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Eta<0         ) MPVsTECP      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Eta>0         ) MPVsTECM      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness<0.04 && tree_Eta>0) MPVsTECP1      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness>0.04 && tree_Eta>0) MPVsTECP2      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness<0.04 && tree_Eta<0) MPVsTECM1      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness>0.04 && tree_Eta<0) MPVsTECM2      ->Fill(         tree_FitMPV);

                                                MPVError      ->Fill(         tree_FitMPVErr);    
                                                MPVErrorVsMPV ->Fill(tree_FitMPV,tree_FitMPVErr);
                                                MPVErrorVsEta ->Fill(tree_Eta,tree_FitMPVErr);
                                                MPVErrorVsPhi ->Fill(tree_Phi,tree_FitMPVErr);
                                                MPVErrorVsN   ->Fill(tree_NEntries,tree_FitMPVErr);
      }

      if(tree_SubDet==1                       ) ChargePIB  ->Add(Proj,1);
      if(tree_SubDet==2                       ) ChargePIE  ->Add(Proj,1);
      if(tree_SubDet==3                       ) ChargeTIB  ->Add(Proj,1);
      if(tree_SubDet==4                       ) ChargeTID  ->Add(Proj,1);
      if(tree_SubDet==4 && tree_Eta<0         ) ChargeTIDM ->Add(Proj,1);
      if(tree_SubDet==4 && tree_Eta>0         ) ChargeTIDP ->Add(Proj,1);
      if(tree_SubDet==5                       ) ChargeTOB  ->Add(Proj,1);
      if(tree_SubDet==6                       ) ChargeTEC  ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Thickness<0.04) ChargeTEC1 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Thickness>0.04) ChargeTEC2 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta>0         ) ChargeTECP ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta<0         ) ChargeTECM ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness<0.04) ChargeTECM1 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness>0.04) ChargeTECM2 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness<0.04) ChargeTECP1 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness>0.04) ChargeTECP2 ->Add(Proj,1);

      if(tree_SubDet==1                       ) ChargeAbsPIB  ->Add(ProjAbsolute,1);
      if(tree_SubDet==2                       ) ChargeAbsPIE  ->Add(ProjAbsolute,1);
      if(tree_SubDet==3                       ) ChargeAbsTIB  ->Add(ProjAbsolute,1);
      if(tree_SubDet==4                       ) ChargeAbsTID  ->Add(ProjAbsolute,1);
      if(tree_SubDet==4 && tree_Eta<0         ) ChargeAbsTIDM ->Add(ProjAbsolute,1);
      if(tree_SubDet==4 && tree_Eta>0         ) ChargeAbsTIDP ->Add(ProjAbsolute,1);
      if(tree_SubDet==5                       ) ChargeAbsTOB  ->Add(ProjAbsolute,1);
      if(tree_SubDet==6                       ) ChargeAbsTEC  ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Thickness<0.04) ChargeAbsTEC1 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Thickness>0.04) ChargeAbsTEC2 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta>0         ) ChargeAbsTECP ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta<0         ) ChargeAbsTECM ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness<0.04) ChargeAbsTECM1 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness>0.04) ChargeAbsTECM2 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness<0.04) ChargeAbsTECP1 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness>0.04) ChargeAbsTECP2 ->Add(ProjAbsolute,1);

      if(tree_SubDet==1                       ) DiffWRTPrevGainPIB  ->Fill(tree_Gain/tree_PrevGain);
      if(tree_SubDet==2                       ) DiffWRTPrevGainPIE  ->Fill(tree_Gain/tree_PrevGain);
      if(tree_SubDet==3                       ) DiffWRTPrevGainTIB  ->Fill(tree_Gain/tree_PrevGain);
      if(tree_SubDet==4                       ) DiffWRTPrevGainTID  ->Fill(tree_Gain/tree_PrevGain);
      if(tree_SubDet==5                       ) DiffWRTPrevGainTOB  ->Fill(tree_Gain/tree_PrevGain);
      if(tree_SubDet==6                       ) DiffWRTPrevGainTEC  ->Fill(tree_Gain/tree_PrevGain);

      if(tree_SubDet==1                       ) GainVsPrevGainPIB  ->Fill(tree_PrevGain,tree_Gain);
      if(tree_SubDet==2                       ) GainVsPrevGainPIE  ->Fill(tree_PrevGain,tree_Gain);
      if(tree_SubDet==3                       ) GainVsPrevGainTIB  ->Fill(tree_PrevGain,tree_Gain);
      if(tree_SubDet==4                       ) GainVsPrevGainTID  ->Fill(tree_PrevGain,tree_Gain);
      if(tree_SubDet==5                       ) GainVsPrevGainTOB  ->Fill(tree_PrevGain,tree_Gain);
      if(tree_SubDet==6                       ) GainVsPrevGainTEC  ->Fill(tree_PrevGain,tree_Gain);




      delete Proj;
      delete ProjAbsolute;
   }printf("\n");



   // ######################################################### PRINT OUT APV INFOS #################################
   unsigned int CountAPV_Total    = 0;
   unsigned int CountAPV_NoEntry  = 0;
   unsigned int CountAPV_NoEntryU = 0;
   unsigned int CountAPV_NoGain   = 0;
   unsigned int CountAPV_NoGainU  = 0;
   unsigned int CountAPV_LowGain  = 0;
   unsigned int CountAPV_DiffGain = 0;

   TrackerMap* tkmap = new TrackerMap("  ParticleGain  ");

   pFile = fopen((output + "_MAP.txt").c_str(),"w");
   fprintf(pFile,"#maxValue = 1.5\n");
   fprintf(pFile,"#minValue = 0.5\n");
   fprintf(pFile,"#defaultColor = 1.0,1.0,1.0,0.1\n");
   printf("Looping on the Tree          :");
   double MaxGain=0;  unsigned int previousMod=0;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}     
      t1->GetEntry(ientry);
      if(previousMod>0&&tree_APVId==0){fprintf(pFile,"%i %f\n",previousMod,MaxGain); tkmap->fill(previousMod, MaxGain);  MaxGain=1;  } 
      previousMod = tree_DetId;
      if(fabs(tree_Gain-1.0)>fabs(MaxGain-1))MaxGain=tree_Gain;
   }printf("\n");
   if(previousMod>0){fprintf(pFile,"%i %f\n",previousMod,MaxGain); tkmap->fill(previousMod, MaxGain); }
   fclose(pFile);

   tkmap->setTitle(TextToPrint + " : Module Gain");
   tkmap->save                 (true, 0.7, 1.3, output + "_TKMap_Gain_MECH.png");
   tkmap->reset();    


   pFile = fopen((output + "_LowResponseModule.txt").c_str(),"w");
   fprintf(pFile,"\n\nALL APVs WITH NO ENTRIES (NO RECO CLUSTER ON IT)\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {      
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);  if(tree_SubDet<3)continue;
      CountAPV_Total++;
      if(tree_NEntries==0){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId);  CountAPV_NoEntry++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_NoEntry,CountAPV_Total,(100.0*CountAPV_NoEntry)/CountAPV_Total);


   fprintf(pFile,"\n\nUNMASKED APVs WITH NO ENTRIES (NO RECO CLUSTER ON IT)\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);   if(tree_SubDet<3)continue;
      if(tree_NEntries==0 && !tree_isMasked){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId); tkmap->fill(tree_DetId, 1); CountAPV_NoEntryU++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_NoEntryU,CountAPV_Total,(100.0*CountAPV_NoEntryU)/CountAPV_Total);

   tkmap->setTitle(TextToPrint + " : #Unmasked APV without any cluster");
   tkmap->save                 (true, 1.0, 6.0, output + "_TKMap_NoCluster_MECH.png");
   tkmap->reset();    


   fprintf(pFile,"\n\nALL APVs WITH NO GAIN COMPUTED\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);   if(tree_SubDet<3)continue;
      if(tree_FitMPV<0){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId); CountAPV_NoGain++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_NoGain,CountAPV_Total,(100.0*CountAPV_NoGain)/CountAPV_Total);

   fprintf(pFile,"\n\nUNMASKED APVs WITH NO GAIN COMPUTED\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);   if(tree_SubDet<3)continue;
      if(tree_FitMPV<0 && !tree_isMasked){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId);  tkmap->fill(tree_DetId, 1); CountAPV_NoGainU++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_NoGainU,CountAPV_Total,(100.0*CountAPV_NoGainU)/CountAPV_Total);

   tkmap->setTitle(TextToPrint + " : #Unmasked APV for which no gain was computed");
   tkmap->save                 (true, 1.0, 6.0, output + "_TKMap_NoGain_MECH.png");
   tkmap->reset();


   fprintf(pFile,"\n\nUNMASKED APVs WITH LOW RESPONSE\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);   if(tree_SubDet<3)continue;
      if(tree_FitMPV>0 && tree_FitMPV<220 && !tree_isMasked){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId); tkmap->fill(tree_DetId, 1); CountAPV_LowGain++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_LowGain,CountAPV_Total,(100.0*CountAPV_LowGain)/CountAPV_Total);


   tkmap->setTitle(TextToPrint + " : #Unmasked APV with a gain<0.75");
   tkmap->save                 (true, 1.0, 6.0, output + "_TKMap_LowGain_MECH.png");
   tkmap->reset();


   fprintf(pFile,"\n\nUNMASKED APVs WITH SIGNIFICANT CHANGE OF GAIN VALUE\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);   if(tree_SubDet<3)continue;
      if(tree_FitMPV>0 && !tree_isMasked && (tree_Gain/tree_PrevGain<0.7 || tree_Gain/tree_PrevGain>1.3)){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId); tkmap->fill(tree_DetId, 1); CountAPV_DiffGain++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_DiffGain,CountAPV_Total,(100.0*CountAPV_DiffGain)/CountAPV_Total);
   fclose(pFile);

   tkmap->setTitle(TextToPrint + " : #Unmasked APV with a gain variation > 30\% w.r.t GT");
   tkmap->save                 (true, 1.0, 6.0, output + "_TKMap_GainChange_MECH.png");
   tkmap->reset();


   // ######################################################### TrackerMAP  ONLY          #################################

   printf("Looping on the Tree          :");
   double MaxError=-1;  previousMod=0;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}     
      t1->GetEntry(ientry);
      if(previousMod>0&&tree_APVId==0&&MaxError>=0){tkmap->fill(previousMod, 100.0*MaxError);  MaxError=-1;  } 
      previousMod = tree_DetId;
      if(tree_FitMPV>0 && tree_FitMPVErr/300.0>MaxError)MaxError=tree_FitMPVErr/300.0;
   }printf("\n");
   if(previousMod>0){tkmap->fill(previousMod, 100.0*MaxError); }
   tkmap->setTitle(TextToPrint + " : Error on Module Gain (%)");
   tkmap->save                 (true, 0, 15, output + "_TKMap_GainError_MECH.png");
   tkmap->reset();    



   printf("Looping on the Tree          :");
   double MaxRatio=-1;  previousMod=0;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      if(previousMod>0&&tree_APVId==0&&MaxRatio>=0){tkmap->fill(previousMod, MaxRatio);  MaxRatio=-1;  }
      previousMod = tree_DetId;
      if(tree_FitMPV>0 && fabs((tree_Gain/tree_PrevGain)-1)>MaxRatio)MaxRatio=fabs((tree_Gain/tree_PrevGain)-1);
   }printf("\n");
   if(previousMod>0){tkmap->fill(previousMod, MaxRatio); }
   tkmap->setTitle(TextToPrint + " : | (G2new / G2gt) - 1 | per module");
   tkmap->save                 (true, 0.0, 0.5, output + "_TKMap_GainRatio_MECH.png");
   tkmap->reset();


   // ######################################################### PRINT DISTRIBUTION INFO #################################
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
  
   pFile = fopen((output + "_SubDetector_MPV.txt").c_str(),"w");

   double Results[5]; TF1* landau;

   landau = getLandau(ChargePIB, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargePIB->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"SubDetChargePIB");
   fprintf(pFile,"PIB   MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargePIE, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargePIE->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"SubDetChargePIE");
   fprintf(pFile,"PIE   MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);


   landau = getLandau(ChargeTIB, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTIB->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"SubDetChargeTIB");
   fprintf(pFile,"TIB   MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTIDM, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTIDM->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"SubDetChargeTIDM");
   fprintf(pFile, "TIDM  MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTIDP, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTIDP->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"SubDetChargeTIDP");
   fprintf(pFile, "TIDP  MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTOB, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTOB->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"SubDetChargeTOB");
   fprintf(pFile, "TOB   MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTECP1, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTECP1->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"SubDetChargeTECP1");
   fprintf(pFile, "TECP1 MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTECP2, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTECP2->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"SubDetChargeTECP2");
   fprintf(pFile, "TECP2 MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTECM1, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTECM1->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"SubDetChargeTECM1");
   fprintf(pFile, "TECM1 MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTECM2, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTECM2->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"SubDetChargeTECM2");
   fprintf(pFile, "TECM2 MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   fclose(pFile);
   // ######################################################### MAKE PLOTS #################################


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = MPV_Vs_EtaTEC;                     legend.push_back("TEC");
   Histos[1] = MPV_Vs_EtaTIB;                     legend.push_back("TIB");
   Histos[2] = MPV_Vs_EtaTID;                     legend.push_back("TID");
   Histos[3] = MPV_Vs_EtaTOB;                     legend.push_back("TOB");
   DrawTH2D((TH2D**)Histos,legend, "", "module #eta", "MPV [ADC/mm]", -3.0,3.0, 0,600);
   DrawLegend (Histos,legend,"","P");
   DrawStatBox(Histos,legend,false);
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"MPV_Vs_EtaSubDet");
   delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPV_Vs_PhiTEC;                     legend.push_back("TEC");
    Histos[1] = MPV_Vs_PhiTIB;                     legend.push_back("TIB");
    Histos[2] = MPV_Vs_PhiTID;                     legend.push_back("TID");
    Histos[3] = MPV_Vs_PhiTOB;                     legend.push_back("TOB");
    DrawTH2D((TH2D**)Histos,legend, "", "module #phi", "MPV [ADC/mm]", -3.4,3.4, 0,600);
    DrawLegend(Histos,legend,"","P");
    DrawStatBox(Histos,legend,false);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"MPV_Vs_PhiSubDet");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = NoMPV;                             legend.push_back("NoMPV");
    DrawTH2D((TH2D**)Histos,legend, "", "z (cm)", "R (cms)", 0,0, 0,0);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"NoMPV", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    ChargeDistrib->GetXaxis()->SetNdivisions(5+500);
    Histos[0] = ChargeDistrib;                     legend.push_back("Charge Vs Index");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "APV Index", "Charge [ADC/mm]", 0,0, 0,0);
    c1->SetLogz(true);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"Charge2D", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = Gains;                             legend.push_back("");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Gain", "Number of APVs", 0.5,1.5, 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true);
    SaveCanvas(c1,output,"Gains");

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVs320;                           legend.push_back("320 #mum");
    Histos[1] = MPVs500;                           legend.push_back("500 #mum");
    Histos[2] = MPVs;                              legend.push_back("320 + 500 #mum");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "MPV [ADC/mm]", "Number of APVs", 100,550, 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"MPVs");
    c1->SetLogy(true);
    SaveCanvas(c1,output,"MPVsLog");
    DrawStatBox(Histos,legend,true);
    c1->SetLogy(false);
    SaveCanvas(c1,output,"MPVsAndStat");
    c1->SetLogy(true);
    SaveCanvas(c1,output,"MPVsLogAndStat");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    MPVsTOB->GetXaxis()->SetNdivisions(5+500);
    Histos[0] = MPVsTIB;                           legend.push_back("TIB (320 #mum)");
    Histos[1] = MPVsTID;                           legend.push_back("TID (320 #mum)");
    Histos[2] = MPVsTOB;                           legend.push_back("TOB (500 #mum)");
    Histos[3] = MPVsTEC1;                          legend.push_back("TEC (320 #mum)");
    Histos[4] = MPVsTEC2;                          legend.push_back("TEC (500 #mum)");
//    Histos[5] = MPVs;                              legend.push_back("All");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "MPV [ADC/mm]", "Number of APVs", 100,550, 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"MPVsSubDet");
    c1->SetLogy(true);
    SaveCanvas(c1,output,"MPVsSubDetLog");
    DrawStatBox(Histos,legend,true);
    SaveCanvas(c1,output,"MPVsSubDetLogAndStat");
    c1->SetLogy(false);
    SaveCanvas(c1,output,"MPVsSubDetAndStat");
    delete c1;



    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    MPVsTOB->GetXaxis()->SetNdivisions(5+500);
    Histos[0] = MPVsTECP1;                          legend.push_back("TEC+ (320 #mum)");
    Histos[1] = MPVsTECP2;                          legend.push_back("TEC+ (500 #mum)");
    Histos[2] = MPVsTECM1;                          legend.push_back("TEC- (320 #mum)");
    Histos[3] = MPVsTECM2;                          legend.push_back("TEC- (500 #mum)");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "MPV [ADC/mm]", "Number of APVs", 100,550, 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"MPVsTEC");
    c1->SetLogy(true);
    SaveCanvas(c1,output,"MPVsTECLog");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVError;                          legend.push_back("MPV Error");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Error on MPV [ADC/mm]", "Number of APVs", 0,500, 0,0);
    DrawStatBox(Histos,legend,true);
    c1->SetLogy(true);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"Error");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsMPV;                     legend.push_back("Error Vs MPV");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "MPV [ADC/mm]", "Error on MPV [ADC/mm]", 0,0, 0,0);
    c1->SetLogz(true);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"Error_Vs_MPV", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsEta;                     legend.push_back("Error Vs Eta");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "module #eta", "Error on MPV [ADC/mm]", 0,0, 0,0);
    c1->SetLogz(true);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"Error_Vs_Eta", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsPhi;                     legend.push_back("Error Vs Phi");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "module #phi", "Error on MPV [ADC/mm]", 0,0, 0,0);
    c1->SetLogz(true);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"Error_Vs_Phi", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsN;                       legend.push_back("Error Vs Entries");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "Number of Entries", "Error on MPV [ADC/mm]", 0,0, 0,0);
    c1->SetLogz(true);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"Error_Vs_N", true);
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargePIB;                         legend.push_back("PIB");
    Histos[1] = ChargePIE;                         legend.push_back("PIE");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC/mm]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawPreliminary(TextToPrint);
    TLine* l0p = new TLine(300, 0,300,((TH1*)Histos[0])->GetMaximum()); l0p->SetLineWidth(3); l0p->SetLineStyle(2); l0p->Draw("same");
    SaveCanvas(c1,output,"Charge");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"PixelChargeAndStat");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTEC;                         legend.push_back("TEC");
    Histos[1] = ChargeTIB;                         legend.push_back("TIB");
    Histos[2] = ChargeTID;                         legend.push_back("TID");
    Histos[3] = ChargeTOB;                         legend.push_back("TOB");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC/mm]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawPreliminary(TextToPrint);
    TLine* l0 = new TLine(300, 0,300,((TH1*)Histos[0])->GetMaximum()); l0->SetLineWidth(3); l0->SetLineStyle(2); l0->Draw("same");
    SaveCanvas(c1,output,"Charge");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"ChargeAndStat");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTECP;                        legend.push_back("TEC+");
    Histos[1] = ChargeTECM;                        legend.push_back("TEC-");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC/mm]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"ChargeTECSide");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTEC1;                        legend.push_back("TEC Thin");
    Histos[1] = ChargeTEC2;                        legend.push_back("TEC Thick");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC/mm]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"ChargeTECThickness");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTIDP;                        legend.push_back("TID+");
    Histos[1] = ChargeTIDM;                        legend.push_back("TID-");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC/mm]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"ChargeTIDSide");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsPIB;                      legend.push_back("PIB");
    Histos[1] = ChargeAbsPIE;                      legend.push_back("PIE");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"PixelChargeAbs");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTEC;                      legend.push_back("TEC");
    Histos[1] = ChargeAbsTIB;                      legend.push_back("TIB");
    Histos[2] = ChargeAbsTID;                      legend.push_back("TID");
    Histos[3] = ChargeAbsTOB;                      legend.push_back("TOB");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"ChargeAbs");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTECP;                     legend.push_back("TEC+");
    Histos[1] = ChargeAbsTECM;                     legend.push_back("TEC-");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"ChargeAbsTECSide");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTEC1;                     legend.push_back("TEC Thin");
    Histos[1] = ChargeAbsTEC2;                     legend.push_back("TEC Thick");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"ChargeAbsTECThickness");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTIDP;                     legend.push_back("TID+");
    Histos[1] = ChargeAbsTIDM;                     legend.push_back("TID-");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"ChargeAbsTIDSide");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPV_Vs_PathlengthThin;             legend.push_back("320 #mum");
    Histos[1] = MPV_Vs_PathlengthThick;            legend.push_back("500 #mum");
    DrawSuperposedHistos((TH1**)Histos, legend, "HIST",  "pathlength [mm]", "MPV [ADC/mm]", 0,0 , 230,380);
    DrawLegend(Histos,legend,"","L");
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"MPV_Vs_Path");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPV_Vs_PathlengthTIB;              legend.push_back("TIB (320 #mum)");
    Histos[1] = MPV_Vs_PathlengthTID;              legend.push_back("TID (320 #mum)");
    Histos[2] = MPV_Vs_PathlengthTOB;              legend.push_back("TOB (500 #mum)");
    Histos[3] = MPV_Vs_PathlengthTEC1;             legend.push_back("TEC (320 #mum)");
    Histos[4] = MPV_Vs_PathlengthTEC2;             legend.push_back("TEC (500 #mum)");
    DrawSuperposedHistos((TH1**)Histos, legend, "HIST",  "pathlength [mm]", "MPV [ADC/mm]", 0,0 , 230,380);
    DrawLegend(Histos,legend,"","L");
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"MPV_Vs_PathSubDet");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = DiffWRTPrevGainPIB;                legend.push_back("PIB");
    Histos[1] = DiffWRTPrevGainPIE;                legend.push_back("PIE");
    DrawSuperposedHistos((TH1**)Histos, legend, "HIST",  "New Gain / Previous Gain", "Number of APV", 0.0,2.0 ,0,0);
    DrawLegend(Histos,legend,"","L");
    c1->SetLogy(true);
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"PixelGainDividedPrevGain");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = DiffWRTPrevGainTIB;                legend.push_back("TIB");
    Histos[1] = DiffWRTPrevGainTID;                legend.push_back("TID");
    Histos[2] = DiffWRTPrevGainTOB;                legend.push_back("TOB");
    Histos[3] = DiffWRTPrevGainTEC;                legend.push_back("TEC");
    DrawSuperposedHistos((TH1**)Histos, legend, "HIST",  "New Gain / Previous Gain", "Number of APV", 0.0,2.0 ,0,0);
    DrawLegend(Histos,legend,"","L");
    c1->SetLogy(true);
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    DrawPreliminary(TextToPrint);
    SaveCanvas(c1,output,"GainDividedPrevGain");
    delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = GainVsPrevGainPIB;                 legend.push_back("PIB");
   Histos[1] = GainVsPrevGainPIE;                 legend.push_back("PIE");
   DrawTH2D((TH2D**)Histos,legend, "", "Previous Gain", "New Gain", 0.5,1.8, 0.5,1.8);
   TLine diagonalP(0.5,0.5,1.8,1.8);
   diagonalP.SetLineWidth(3);
   diagonalP.SetLineStyle(2);
   diagonalP.Draw("same");
   DrawLegend (Histos,legend,"","P");
   DrawStatBox(Histos,legend,false);
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"PixelGainVsPrevGain");
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = GainVsPrevGainTEC;                 legend.push_back("TEC");
   Histos[1] = GainVsPrevGainTIB;                 legend.push_back("TIB");
   Histos[2] = GainVsPrevGainTID;                 legend.push_back("TID");
   Histos[3] = GainVsPrevGainTOB;                 legend.push_back("TOB");
   DrawTH2D((TH2D**)Histos,legend, "", "Previous Gain", "New Gain", 0.5,1.8, 0.5,1.8);
   TLine diagonal(0.5,0.5,1.8,1.8);
   diagonal.SetLineWidth(3);
   diagonal.SetLineStyle(2);
   diagonal.Draw("same");
   DrawLegend (Histos,legend,"","P");
   DrawStatBox(Histos,legend,false);
   DrawPreliminary(TextToPrint);
   SaveCanvas(c1,output,"GainVsPrevGain");
   delete c1;


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

TH1D* ChargeToMPV(TH2* InputHisto, string Name,  bool DivideByX)
{
   TH1D* toReturn = new TH1D(Name.c_str(),Name.c_str(),InputHisto->GetXaxis()->GetNbins(), InputHisto->GetXaxis()->GetXmin(), InputHisto->GetXaxis()->GetXmax() );
   double Results[5];

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

