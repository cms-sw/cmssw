#define JetHij_cxx
#include "JetHij.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void JetHij::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L JetHij.C
//      Root > JetHij t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
//
   Double_t M_PI = 3.14159265358979323846;
   Double_t M_PI2 = 3.14159265358979323846*2.;
//===> Histograms
// CENTRALITY_BIN
   TH1F *h_centrality   = new TH1F("h_centrality","",100,0., 100.);

// Mean ETGEN
   TH1F *h_Etgen_75_85 = new TH1F("h_Etgen_75_85","",70, 60., 90.);
   TH1F *h_Etgen_85_95 = new TH1F("h_Etgen_85_95","",70, 70., 100.);
   TH1F *h_Etgen_95_105 = new TH1F("h_Etgen_95_105","",70, 80., 110.);
   TH1F *h_Etgen_105_115 = new TH1F("h_Etgen_105_115","",70, 90., 120.);

// Etrec/ETgen - reconstructed from calotowers
   TH1F *h_EtrecEtgen_75_85 = new TH1F("h_EtrecEtgen_75_85","",70, 0., 2.);
   TH1F *h_EtrecEtgen_85_95 = new TH1F("h_EtrecEtgen_85_95","",70, 0., 2.);
   TH1F *h_EtrecEtgen_95_105 = new TH1F("h_EtrecEtgen_95_105","",70, 0., 2.);
   TH1F *h_EtrecEtgen_105_115 = new TH1F("h_EtrecEtgen_105_115","",70, 0., 2.);

// Etcorzsjpt/ETgen - jpt + ZS
   TH1F *h_EtcorzsjptEtgen_75_85 = new TH1F("h_EtcorzsjptEtgen_75_85","",70, 0., 2.);
   TH1F *h_EtcorzsjptEtgen_85_95 = new TH1F("h_EtcorzsjptEtgen_85_95","",70, 0., 2.);
   TH1F *h_EtcorzsjptEtgen_95_105 = new TH1F("h_EtcorzsjptEtgen_95_105","",70, 0., 2.);
   TH1F *h_EtcorzsjptEtgen_105_115 = new TH1F("h_EtcorzsjptEtgen_105_115","",70, 0., 2.);

//===>

   Int_t ifile = 0;  

   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);

//       cout <<" Event= "<<ientry<<endl;
      if(ientry == 0) {
        cout<<" New file "<<ifile<<endl;
        ifile++;
      }

      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;

//===>
         Float_t centrality = centrality_bin;

      if( NumGenJets == 0 ) continue;
      if( NumRecoJetsCaloTower == 0  )  continue;
      if( NumRecoJetsJPTCorrected2 == 0  )  continue;

      Int_t njets = NumRecoJetsCaloTower;
      for( Int_t myjet=0;myjet<njets;myjet++)
      {
      //
//  Matching with Genjets
//
     Int_t firstgen = -1;

     for(Int_t i=0; i<NumGenJets; i++)
     {

         Float_t deta2 = JetGenEta[i]-JetRecoEtaCaloTower[myjet];
         Float_t dphi2 = fabs(JetGenPhi[i]-JetRecoPhiCaloTower[myjet]);
         if( dphi2 > M_PI ) dphi2 = M_PI2 - dphi2;
         Float_t dr0 = sqrt(dphi2*dphi2+deta2*deta2);
         if( dr0 < 0.3 ) firstgen = i;

     } // Cycle on GenJets

    if( firstgen < 0 ) continue;

//
// Matching with JPT+ZS corrected jets
//
     Int_t firstcor = -1;

     cout<<"NumRecoJetsJPTCorrected2="<<NumRecoJetsJPTCorrected2<<endl;

     for(Int_t i=0; i<NumRecoJetsJPTCorrected2; i++)
     {
         Float_t deta2 = JetRecoEtaJPTCorrected2[i]-JetRecoEtaCaloTower[myjet];
         Float_t dphi2 = fabs(JetRecoPhiJPTCorrected2[i]-JetRecoPhiCaloTower[myjet]);
         if( dphi2 > M_PI ) dphi2 = M_PI2 - dphi2;
         Float_t dr0 = sqrt(dphi2*dphi2+deta2*deta2);
         if( dr0 < 0.3 ) firstcor = i;

     } // Cycle on CorrJets
     if( firstcor < 0 ) continue;
// Filling histograms
     //cout<<"firstgen="<< firstgen<<" firstcor= "<<firstcor<<endl;
     Float_t etgen,etagen,etcalo,etcorzsjpt;
     etcalo = JetRecoEtCaloTower[myjet];
     etgen = JetGenEt[firstgen];
     etagen = JetGenEta[firstgen];
     etcorzsjpt = JetRecoEtJPTCorrected2[firstcor];

     Float_t ratio = etcalo/etgen;
     Float_t ratio_corzsjpt = etcorzsjpt/etgen;

     if(fabs(etagen)>1.4) continue;

         if(75.<etgen&& etgen<85.) {
           h_EtrecEtgen_75_85->Fill(ratio);
//           h_EtcorzsEtgen_75_85->Fill(ratio_corzs);
           h_EtcorzsjptEtgen_75_85->Fill(ratio_corzsjpt);
         }
         if(85.<etgen&& etgen<95.) {
           h_EtrecEtgen_85_95->Fill(ratio);
//           h_EtcorzsEtgen_85_95->Fill(ratio_corzs);
           h_EtcorzsjptEtgen_85_95->Fill(ratio_corzsjpt);
         }
         if(95.<etgen&& etgen<105.) {
           h_EtrecEtgen_95_105->Fill(ratio);
//           h_EtcorzsEtgen_95_105->Fill(ratio_corzs);
           h_EtcorzsjptEtgen_95_105->Fill(ratio_corzsjpt);
         }
         if(105.<etgen&& etgen<115.) {
           h_EtrecEtgen_105_115->Fill(ratio);
//           h_EtcorzsEtgen_105_115->Fill(ratio_corzs);
           h_EtcorzsjptEtgen_105_115->Fill(ratio_corzsjpt);
         }

    } // calojets
//  } // Nevents

//===>
         h_centrality       -> Fill(centrality);

   } // Nevents

     TFile efile("histo.root","recreate");

     h_centrality       ->  Write();

// Gen and rec mean
    h_Etgen_75_85->Write();
    h_Etgen_85_95->Write();
    h_Etgen_95_105->Write();
    h_Etgen_105_115->Write();

    h_EtrecEtgen_75_85->Write();
    h_EtrecEtgen_85_95->Write();
    h_EtrecEtgen_95_105->Write();
    h_EtrecEtgen_105_115->Write();

    h_EtcorzsjptEtgen_75_85->Write();
    h_EtcorzsjptEtgen_85_95->Write();
    h_EtcorzsjptEtgen_95_105->Write();
    h_EtcorzsjptEtgen_105_115->Write();

          efile.Close();

}
