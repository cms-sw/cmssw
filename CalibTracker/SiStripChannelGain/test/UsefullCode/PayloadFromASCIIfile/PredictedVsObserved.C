

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
#include<tdrstyle.C>

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


void PredictedVsObserved(){

   //read gain modified from a txt file
   std::map<unsigned int, double> gainModifiers;
   FILE* inFile = fopen("GainRatio.txt","r");
   char line[4096];
   while(fgets(line, 4096, inFile)!=NULL){
      unsigned int detId;  double modifier;
      sscanf(line, "%u %lf", &detId, &modifier);
//      printf("(%i , %f) \t", detId, modifier);
      gainModifiers[detId] = modifier;
   }
   fclose(inFile);


   string inputFileTree = "../../Data_Run_251252_to_251252_PCL/Gains_Tree.root"; 
   string moduleName = "SiStripCalib";

   unsigned int  tree1_Index;
   unsigned int  tree1_DetId;
   unsigned char tree1_APVId;
   double        tree1_Gain;
   double        tree1_PrevGain;
   double        tree1_NEntries;

   TFile* f1     = new TFile(inputFileTree.c_str());
   TTree *t1     = (TTree*)f1->Get((moduleName+"/APVGain").c_str());


   t1->SetBranchAddress("Index"             ,&tree1_Index      );
   t1->SetBranchAddress("DetId"             ,&tree1_DetId      );
   t1->SetBranchAddress("APVId"             ,&tree1_APVId      );
   t1->SetBranchAddress("Gain"              ,&tree1_Gain       );
   t1->SetBranchAddress("PrevGain"          ,&tree1_PrevGain   );
   t1->SetBranchAddress("NEntries"          ,&tree1_NEntries   );



   string inputFileTree2 = "../../Data_Run_252126_to_252226_CalibTree/Gains_Tree.root";

   unsigned int  tree2_Index;
   unsigned int  tree2_DetId;
   unsigned char tree2_APVId;
   double        tree2_Gain;
   double        tree2_PrevGain;
   double        tree2_NEntries;


   TFile* f2     = new TFile(inputFileTree2.c_str());
   TTree *t2     = (TTree*)f2->Get((moduleName+"/APVGain").c_str());


   t2->SetBranchAddress("Index"             ,&tree2_Index      );
   t2->SetBranchAddress("DetId"             ,&tree2_DetId      );
   t2->SetBranchAddress("APVId"             ,&tree2_APVId      );
   t2->SetBranchAddress("Gain"              ,&tree2_Gain       );
   t2->SetBranchAddress("PrevGain"          ,&tree2_PrevGain   );
   t2->SetBranchAddress("NEntries"          ,&tree2_NEntries   );



   TH1D*  HRatio = new TH1D("Ratio", "Ratio", 150, 0.5, 1.5);
   TH1D*  HRatioB = new TH1D("RatioB", "RatioB", 150, 0.5, 1.5);
   TH1D*  HRatioC = new TH1D("RatioC", "RatioC", 150, 0.5, 1.5);
   TH2D*  HVS    = new TH2D("VS", "VS", 150, 0.5, 1.5, 150, 0.5, 1.5);
   

   TrackerMap* tkmap = new TrackerMap("  Gain : Measured(R252116+R252226) / Predicted(R251252*StrenghRatio from DelayScan)");

   FILE* pFile = fopen("Gains_ASCIIObserved.txt","w");

   int SAME=0;
   int DIFF=0;

   int PreviousId = -1;
   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on the Tree          :");
   int TreeStep = t1->GetEntries()/50;if(TreeStep==0)TreeStep=1;
   std::vector<double> gains;
   double MEAN = 0;  int NAPV=0;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      t2->GetEntry(ientry);

//     if(tree1_DetId==369153064){
     if(tree1_DetId==402673324){
         printf("%i | %i : %f --> %f  (%f)\n", tree1_DetId, tree1_APVId, tree1_PrevGain, tree1_Gain, tree1_NEntries);
         printf("%i | %i : %f --> %f  (%f)\n", tree2_DetId, tree2_APVId, tree2_PrevGain, tree2_Gain, tree2_NEntries);
         printf("\n");
      }

      
       if(fabs(tree2_PrevGain - tree1_Gain) > 0.05*tree2_PrevGain){
//         printf("%i | %i : %f != %f | %i | %i\n", tree1_DetId, tree1_APVId, tree1_Gain, tree2_PrevGain, tree2_APVId, tree2_DetId);
         DIFF++;
      }else{
         SAME++;
      }


      if((tree1_APVId==0 && PreviousId>0) || ientry==t1->GetEntries()-1){  //new module or very last APV
         fprintf(pFile,"%i ",PreviousId); for(unsigned int i=0;i<gains.size();i++){fprintf(pFile,"%f ", gains[i]);}   fprintf(pFile, "\n");

          if(NAPV>0)tkmap->fill(tree1_DetId, MEAN/NAPV);      
          MEAN = 0; NAPV=0;
         gains.clear();
      }
      PreviousId = tree1_DetId;

      if(tree2_Gain==1.0000)continue;

      double modifier = 1.0;
      if(gainModifiers.find(tree1_DetId)!=gainModifiers.end()){modifier = gainModifiers[tree1_DetId]; }
//      gains.push_back(tree1_Gain * modifier);
      gains.push_back(tree2_Gain / (tree1_Gain * modifier) );
       

      HRatio->Fill(tree2_Gain / (tree1_Gain * modifier));
      if((fabs(tree2_PrevGain-tree1_Gain*modifier)/tree2_PrevGain)>0.03)  HRatioB->Fill(tree2_Gain / (tree1_Gain * modifier));
      if((fabs(tree2_PrevGain-tree1_Gain*modifier)/tree2_PrevGain)<0.03){ 
         HRatioC->Fill(tree2_Gain / (tree1_Gain * modifier));
         MEAN += tree2_Gain / (tree1_Gain * modifier);
         NAPV++;
         HVS   ->Fill((tree1_Gain * modifier), tree2_Gain);
      }

   }printf("\n");
   fclose(pFile);

   printf("%i (same) + %i (diff) = %i\n", SAME, DIFF, SAME+DIFF);

//   tkmap->setTitle("Gain: Predicted/Current    (white=unmodified)");
   tkmap->save(true, 0.95, 1.1, "tkMapPredictedVsObserved.png");
   tkmap->reset();    




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






if(true){
   TCanvas* c1 = new TCanvas("c1","c1,",900,600);

   HRatio->SetLineWidth(2);
   HRatio->SetLineColor(1);
   HRatioB->SetLineWidth(2);
   HRatioB->SetLineColor(2);

   TH1D* frame = HRatio;
   frame->GetXaxis()->SetNdivisions(505);
   frame->SetTitle("");
   frame->SetStats(kFALSE);
//   frame->GetXaxis()->SetTitle("Integrated Lumi (fb^{-1})");
     frame->GetXaxis()->SetTitle("Gain : Measured / Predicted");
//   frame->GetYaxis()->SetTitle("Average Gain");
//   frame->SetMaximum(1.25);
//   frame->SetMinimum(0.80);
   frame->GetYaxis()->SetTitle("#APV");
//   frame->SetMaximum(1.10);
//   frame->SetMinimum(0.85);
   frame->GetYaxis()->SetTitleOffset(1.50);
   frame->Draw("");

   HRatioB->Draw("same");
//   c1->SetLogy(true);


   TLegend* leg = new TLegend(0.15,0.93,0.80, 0.75);
   leg->SetFillStyle(0);
   leg->SetBorderSize(0);
   leg->SetTextFont(43);
   leg->SetTextSize(20);
   leg->SetNColumns(1);
   leg->Clear();
   leg->AddEntry(HRatio, "All APVs", "L");
   leg->AddEntry(HRatioB, "APVs with bugged predicted G2", "L");
   leg->Draw();


   c1->SaveAs("Ratio.png");

   HRatioC->SaveAs("histo.root");
}



if(true){
   TCanvas* c1 = new TCanvas("c1","c1,",900,600);

   TH2D* frame = HVS;
   frame->GetXaxis()->SetNdivisions(505);
   frame->SetTitle("");
   frame->SetStats(kFALSE);
//   frame->GetXaxis()->SetTitle("Integrated Lumi (fb^{-1})");
     frame->GetXaxis()->SetTitle("Gain: Predicted(R251252*StrenghRatio from DelayScan)");
//   frame->GetYaxis()->SetTitle("Average Gain");
//   frame->SetMaximum(1.25);
//   frame->SetMinimum(0.80);
   frame->GetYaxis()->SetTitle("Gain : Measured(R252116+R252226)");
//   frame->SetMaximum(1.10);
//   frame->SetMinimum(0.85);
   frame->GetYaxis()->SetTitleOffset(1.50);
   frame->Draw("COLZ");
   c1->SetLogz(true);
   c1->SaveAs("SCATTER.png");
}


}
