

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TTree.h"
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

#include<vector>

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"

using namespace fwlite;
using namespace std;
using namespace edm;
#endif




void tkMap(string TextToPrint_="CMS Preliminary 2015"){

}


void MakeAsciiFileFromTree(){
   string inputFileTree = "../../Data_Run_251252_to_251252_PCL/Gains_Tree.root"; 
   string moduleName = "SiStripCalib";

   unsigned int  tree1_Index;
   unsigned int  tree1_DetId;
   unsigned char tree1_APVId;
   double        tree1_Gain;
   double        tree1_PrevGain;

   TFile* f1     = new TFile(inputFileTree.c_str());
//   TTree *t1     = (TTree*)GetObjectFromPath(f1,moduleName+"/APVGain");
   TTree *t1     = (TTree*)f1->Get((moduleName+"/APVGain").c_str());


   t1->SetBranchAddress("Index"             ,&tree1_Index      );
   t1->SetBranchAddress("DetId"             ,&tree1_DetId      );
   t1->SetBranchAddress("APVId"             ,&tree1_APVId      );
   t1->SetBranchAddress("Gain"              ,&tree1_Gain       );
   t1->SetBranchAddress("PrevGain"          ,&tree1_PrevGain   );



   //read gain modified from a txt file
   std::map<unsigned int, double> gainModifiers;
   FILE* inFile = fopen("GainRatio.txt","r");
   char line[4096];
   while(fgets(line, 4096, inFile)!=NULL){
      unsigned int detId;  double modifier;
      sscanf(line, "%u %lf", &detId, &modifier);
      printf("(%i , %f) \t", detId, modifier);
      gainModifiers[detId] = modifier;
   }
   fclose(inFile);
   

   TrackerMap* tkmap = new TrackerMap("  ParticleGain  ");

   FILE* pFile = fopen("Gains_ASCII.txt","w");

   int PreviousId = -1;
   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on the Tree          :");
   int TreeStep = t1->GetEntries()/50;if(TreeStep==0)TreeStep=1;
   std::vector<double> gains;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);

      if((tree1_APVId==0 && PreviousId>0) || ientry==t1->GetEntries()-1){  //new module or very last APV
         fprintf(pFile,"%i ",PreviousId); for(unsigned int i=0;i<gains.size();i++){fprintf(pFile,"%f ", gains[i]);}   fprintf(pFile, "\n");
         gains.clear();
      }
      PreviousId = tree1_DetId;

      double modifier = 1.0;
      if(gainModifiers.find(tree1_DetId)!=gainModifiers.end()){modifier = gainModifiers[tree1_DetId]; }
      gains.push_back(tree1_Gain * modifier);

      if(tree1_APVId==0 and modifier!=1.000){tkmap->fill(tree1_DetId, modifier); }

   }printf("\n");
   fclose(pFile);

   tkmap->setTitle("Gain: Predicted/Current    (white=unmodified)");
   tkmap->save(true, 1.0, 1.2, "tkMap.png");
   tkmap->reset();    

}
