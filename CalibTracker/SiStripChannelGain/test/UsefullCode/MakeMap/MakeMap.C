#include <vector>

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


#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

void MakeMap()
{
   double P_Min               = 1;
   double P_Max               = 15;
   int    P_NBins             = 14  ;
   double Path_Min            = 0.2 ;
   double Path_Max            = 1.6 ;
   int    Path_NBins          = 42  ;
   double Charge_Min          = 0   ;
   double Charge_Max          = 5000;
   int    Charge_NBins        = 500 ;

   TH3F* Charge_Vs_Path = new TH3F ("Charge_Vs_Path"     , "Charge_Vs_Path" , P_NBins, P_Min, P_Max, Path_NBins, Path_Min, Path_Max, Charge_NBins, Charge_Min, Charge_Max);



   ///////////////////////////////////////// MAKE MAP OF GAINS
   std::map<unsigned int,double> Gains;
   TChain* t1 = new TChain("SiStripCalib/APVGain");
   t1->Add("file:Gains.root");

   unsigned int  tree_DetId;
   unsigned char tree_APVId;
   double        tree_Gain;

   t1->SetBranchAddress("DetId"             ,&tree_DetId      );
   t1->SetBranchAddress("APVId"             ,&tree_APVId      );
   t1->SetBranchAddress("Gain"              ,&tree_Gain       );

   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
       t1->GetEntry(ientry);
       Gains[tree_DetId<<3 | tree_APVId] = tree_Gain;
   }
   /////////////////////////////////////////

   vector<string> VInputFiles;
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190645.root");       //size = 1727.02MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190646.root");       //size = 1863.46MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190659.root");       //size = 2661.59MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190661.root");       //size = 6034.83MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190663.root");       //size = 1668.16MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190678.root");       //size = 9927.4MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190679.root");       //size = 913.426MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190688.root");       //size = 4128.89MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190702.root");       //size = 1996.21MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190703.root");       //size = 4408.31MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190705.root");       //size = 7187.65MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190706.root");       //size = 2186.41MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190707.root");       //size = 5083.74MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190708.root");       //size = 2683.04MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190733.root");       //size = 7182.07MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190736.root");       //size = 3222.12MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190738.root");       //size = 5912.86MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190782.root");       //size = 7974.95MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190895.root");       //size = 15331.2MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190906.root");       //size = 7652.26MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190945.root");       //size = 1562.03MB
//   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_190949.root");       //size = 8016.04MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191046.root");       //size = 4838.85MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191056.root");       //size = 97.7042MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191057.root");       //size = 1283.01MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191062.root");       //size = 10698.4MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191086.root");       //size = 9208.68MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191090.root");       //size = 2663.71MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191201.root");       //size = 899.183MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191202.root");       //size = 1975.63MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191226.root");       //size = 30174.2MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191247.root");       //size = 23016.2MB
//   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191248.root");       //size = 1653.05MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191264.root");       //size = 2810.84MB
//   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191271.root");       //size = 6688.46MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191276.root");       //size = 256.78MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191277.root");       //size = 16743.7MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191367.root");       //size = 56.7291MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191691.root");       //size = 193.001MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191692.root");       //size = 522.558MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191694.root");       //size = 849.879MB
//   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191700.root");       //size = 12765.5MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191701.root");       //size = 2647.48MB
//   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191718.root");       //size = 3851.63MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191720.root");       //size = 3203.27MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191721.root");       //size = 3108.56MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191726.root");       //size = 171.324MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191800.root");       //size = 901.921MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191810.root");       //size = 1926.16MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191811.root");       //size = 2952.6MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191830.root");       //size = 7499.53MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191833.root");       //size = 2139.62MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191834.root");       //size = 7184.75MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191837.root");       //size = 1212.49MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191839.root");       //size = 653.932MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191842.root");       //size = 264.614MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191845.root");       //size = 460.653MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191849.root");       //size = 877.692MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191856.root");       //size = 2393.8MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191857.root");       //size = 483.061MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191858.root");       //size = 809.952MB
   VInputFiles.push_back("rfio:/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12//calibTree_191859.root");       //size = 2158.63MB


   TString EventPrefix("");
   TString EventSuffix("");

   TString TrackPrefix("track");
   TString TrackSuffix("");

   TString CalibPrefix("GainCalibration");
   TString CalibSuffix("");
 

   for(unsigned int i=0;i<VInputFiles.size();i++){
      printf("Openning file %3i/%3i --> %s\n",i+1, (int)VInputFiles.size(), (char*)(VInputFiles[i].c_str())); fflush(stdout);
      TChain* tree = new TChain("gainCalibrationTree/tree");
      tree->Add(VInputFiles[i].c_str());

      unsigned int                 eventnumber    = 0;    tree->SetBranchAddress(EventPrefix + "event"          + EventSuffix, &eventnumber   , NULL);
      unsigned int                 runnumber      = 0;    tree->SetBranchAddress(EventPrefix + "run"            + EventSuffix, &runnumber     , NULL);
      std::vector<bool>*           trigtech       = 0;    tree->SetBranchAddress(EventPrefix + "TrigTech"       + EventSuffix, &trigtech      , NULL); 

      std::vector<double>*         trackchi2ndof  = 0;    tree->SetBranchAddress(TrackPrefix + "chi2ndof"       + TrackSuffix, &trackchi2ndof , NULL);
      std::vector<float>*          trackp         = 0;    tree->SetBranchAddress(TrackPrefix + "momentum"       + TrackSuffix, &trackp        , NULL);
      std::vector<float>*          trackpt        = 0;    tree->SetBranchAddress(TrackPrefix + "pt"             + TrackSuffix, &trackpt       , NULL);
      std::vector<unsigned int>*   trackhitsvalid = 0;    tree->SetBranchAddress(TrackPrefix + "hitsvalid"      + TrackSuffix, &trackhitsvalid, NULL);

      std::vector<int>*            trackindex     = 0;    tree->SetBranchAddress(CalibPrefix + "trackindex"     + CalibSuffix, &trackindex    , NULL);
      std::vector<unsigned int>*   rawid          = 0;    tree->SetBranchAddress(CalibPrefix + "rawid"          + CalibSuffix, &rawid         , NULL);
      std::vector<unsigned short>* firststrip     = 0;    tree->SetBranchAddress(CalibPrefix + "firststrip"     + CalibSuffix, &firststrip    , NULL);
      std::vector<unsigned short>* nstrips        = 0;    tree->SetBranchAddress(CalibPrefix + "nstrips"        + CalibSuffix, &nstrips       , NULL);
      std::vector<unsigned int>*   charge         = 0;    tree->SetBranchAddress(CalibPrefix + "charge"         + CalibSuffix, &charge        , NULL);
      std::vector<float>*          path           = 0;    tree->SetBranchAddress(CalibPrefix + "path"           + CalibSuffix, &path          , NULL);
      std::vector<unsigned char>*  amplitude      = 0;    tree->SetBranchAddress(CalibPrefix + "amplitude"      + CalibSuffix, &amplitude     , NULL);

      int TreeStep = tree->GetEntries()/50;if(TreeStep==0)TreeStep=1;
      for (unsigned int ientry = 0; ientry < tree->GetEntries(); ientry++) {
         if(ientry%TreeStep==0){printf(".");fflush(stdout);}
         tree->GetEntry(ientry);

         int FirstAmplitude = 0;
         for(unsigned int c=0;c<(*path).size();c++){
            FirstAmplitude+=(*nstrips)[c];
            int t = (*trackindex)[c];
            if((*trackpt)[t]<5)continue;
            if((*trackhitsvalid)[t]<5)continue;

            int Charge = 0;
            for(unsigned int s=0;s<(*nstrips)[c];s++){
              int StripCharge =  (*amplitude)[FirstAmplitude-(*nstrips)[c]+s];
              if(StripCharge<254){
                 StripCharge=(int)(StripCharge/Gains[(*rawid)[c]<<3 | (*firststrip)[c]/128]);
                 if(StripCharge>=1024){
                    StripCharge = 255;
                 }else if(StripCharge>=254){
                    StripCharge = 254;
                 }
              }
              Charge += StripCharge;
            }
//          printf("ChargeDifference = %i Vs %i with Gain = %f\n",(*charge)[c],Charge,Gains[(*rawid)[c]]);
            double ClusterChargeOverPath   =  ( (double) Charge )/(*path)[c] ;       



	    SiStripDetId SSdetId((*rawid)[c]);
            //printf("ModuleGeometry = %i\n",SSdetId.moduleGeometry());




//            Charge_Vs_Path->Fill((*trackp)[t],(*path)[c],ClusterChargeOverPath);
            Charge_Vs_Path->Fill(SSdetId.moduleGeometry(),(*path)[c],ClusterChargeOverPath);
         }
      }printf("\n");
   }printf("\n");

   TFile * out = new TFile("ProbaMap.root", "RECREATE");
   Charge_Vs_Path->Write();
   out->Write();
   out->Close();
   delete out;

}






