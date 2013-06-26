#define MinBias_cxx
#include "MinBias.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void MinBias::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L MinBias.C
//      Root > MinBias t
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
   TH1F  *hCalo1[110][50];
   TH1F  *hCalo2[110][50];
   
    for(int i=0;i<100;i++){
    char str0[6];
    char str1[6];
    for(int j=0;j<50;j++){
    
    int k = i*10000+j;
    sprintf(str0,"enpl%d",k);
    sprintf(str1,"enmin%d",k);
   
    hCalo1[i][j] = new TH1F(str0, "enpl", 300, 0.1665, 100.1665); 
    hCalo2[i][j] = new TH1F(str1, "enmin", 300, 0.1665, 100.1665);
    }
    }

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
//      if (fabs(eta)<0.5) continue;
//      cout<<" mom1 "<<iphi<<" "<<ieta<<endl;
      if(ieta<0) hCalo1[iphi][abs(ieta)]->Fill(mom1);
      if(ieta>=0) hCalo2[iphi][ieta]->Fill(mom1);
      if ( ieta == -2 && iphi == 35 ) cout<<" "<<mom1<<endl;
      if ( ieta == 2 && iphi == 35 ) cout<<" "<<mom1<<endl;
      if ( mom1 > 1. ) cout<<mom1<<endl;
      
   }
   hCalo1[35][2]->Draw();
}
