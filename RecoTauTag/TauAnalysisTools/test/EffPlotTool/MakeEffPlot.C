#define MakeEffPlot_cxx
#include "MakeEffPlot.h"
#include <TH2.h>
#include <TH1F.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>


void MakeEffPlot::Loop()
{

   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;


//-----------------------------------------------------------//
//                                                           //
// Example of a efficiency calculation for filter hltPFTau35 //
//                                                           // 
//-----------------------------------------------------------//

   TCanvas *TrigEff = new TCanvas("TrigEff","TrigEff");
   TrigEff->SetFillColor(10);


   Int_t RebinFak=10; 

   TH1F* h1_Tag_pt = new TH1F("h1_Tag_pt", "h1_Tag_pt", 1000, 0., 1000.);
   TH1F* h1_Match_pt = new TH1F("h1_Match_pt", "h1_Match_pt", 1000, 0., 1000.);

   // Loop over the efficiency tree
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
    

      if(tagTauPt>35.){ // test tag tau for trigger requirement  

        h1_Tag_pt->Fill(tagTauPt);

        if(hltPFTau35){ // test if trigger object is matched to tag tau
              h1_Match_pt->Fill(tagTauPt);
         }
      }  
   }


   h1_Match_pt->Rebin(RebinFak);
   h1_Tag_pt->Rebin(RebinFak);

   h1_Match_pt->Divide(h1_Tag_pt); // calculate efficiency distribution


   //plot efficiency distribution
   TrigEff->cd();

   h1_Match_pt->GetXaxis()->SetTitle("RecoTau_p_{T} [GeV]");
   

   h1_Match_pt->Draw("hist");
   TrigEff->Print("EffPlot.pdf");

}

