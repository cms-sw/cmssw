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


void KeepOnlyGain()
{
   TChain* tree = new TChain("SiStripCalib/APVGain");
   tree->Add("Gains_Tree.root");


   TFile* out_file = new TFile("Gains.root","recreate");
   TDirectory* out_dir = out_file->mkdir("SiStripCalib","SiStripCalib");
   out_dir->cd();


   TTree* out_tree = tree->CloneTree(0);
   out_tree->SetName("APVGain");
   out_tree->Write();

   int TreeStep = tree->GetEntries()/50;if(TreeStep==0)TreeStep=1;
   for (unsigned int ientry = 0; ientry < tree->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      tree->GetEntry(ientry);
      out_tree->Fill();
   }printf("\n");

   out_file->Write();
   out_file->Close();

}






