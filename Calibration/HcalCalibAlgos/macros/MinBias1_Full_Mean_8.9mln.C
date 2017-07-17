#define MinBias1_cxx
#include "MinBias1.h_8.9mln"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream.h>
#include <fstream>
#include <sstream>

void MinBias1::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L MinBias1.C
//      Root > MinBias1 t
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

   Int_t nbytes = 0, nb = 0;
   
   FILE *Out1 = fopen("mean_minbias_8.9mln.txt", "w+");
   FILE *Out2 = fopen("mean_noise_8.9mln.txt", "w+");
      
	 Float_t minmean_MB[43][73][5][5];
	 Float_t minmean_NS[43][73][5][5];
	 Float_t minmean_Diff[43][73][5][5];
	 Int_t minnevetaphi[43][73][5][5];
	 
         Float_t minerr_MB[43][73][5][5];
         Float_t minerr_NS[43][73][5][5];
         Float_t minerr_Diff[43][73][5][5];
	 
	 Float_t plmean_MB[43][73][5][5];
	 Float_t plmean_NS[43][73][5][5];
	 Float_t plmean_Diff[43][73][5][5];
	 Int_t plnevetaphi[43][73][5][5];
	 
         Float_t plerr_MB[43][73][5][5];
         Float_t plerr_NS[43][73][5][5];
         Float_t plerr_Diff[43][73][5][5];
         Int_t mysubdetpl0[43][73][5][5];
         Int_t mysubdetmin0[43][73][5][5];
 
  for(Int_t ietak = 0; ietak < 43; ietak++ )
  {
   for(Int_t idep = 0; idep < 5; idep++ )
   {
    for(Int_t isub = 0; isub < 5; isub++ )
   {
     for(Int_t iphik = 0; iphik < 73; iphik++ )
    {

	 minmean_MB[ietak][iphik][idep][isub] = 0.;
	 minmean_NS[ietak][iphik][idep][isub] = 0.;
	 minmean_Diff[ietak][iphik][idep][isub] = 0.;
	 minnevetaphi[ietak][iphik][idep][isub] = 0;
	 
         minerr_MB[ietak][iphik][idep][isub] = 0.;
         minerr_NS[ietak][iphik][idep][isub] = 0.;
         minerr_Diff[ietak][iphik][idep][isub] = 0.;
	 
	 plmean_MB[ietak][iphik][idep][isub] = 0.;
	 plmean_NS[ietak][iphik][idep][isub] = 0.;
	 plmean_Diff[ietak][iphik][idep][isub] = 0.;
	 plnevetaphi[ietak][iphik][idep][isub] = 0;
	 
         plerr_MB[ietak][iphik][idep][isub] = 0.;
         plerr_NS[ietak][iphik][idep][isub] = 0.;
         plerr_Diff[ietak][iphik][idep][isub] = 0.;

    }
    } 
   }
  }

   TH1F* h1noise = new TH1F ("h1noise","h1noise",100, -0.1,0.4);
   TH1F* h1noise_mean_1 = new TH1F ("h1noise_mean_1","h1noise_mean_1",100, -0.1,0.4); 
   TH1F* h1noise_mean_2 = new TH1F ("h1noise_mean_2","h1noise_mean_2",100, -0.1,0.4); 
   TH1F* h1noise_mean_3 = new TH1F ("h1noise_mean_3","h1noise_mean_3",100, -0.1,0.4); 
   TH1F* h1noise_mean_4 = new TH1F ("h1noise_mean_4","h1noise_mean_4",100, -0.1,0.4); 
   TH1F* h1noise_mean_6 = new TH1F ("h1noise_mean_6","h1noise_mean_6",100, -0.1,0.4); 
  
   TH2F* hlnoise = new TH2F ("hlnoise","hlnoise",28, -14, 14, 72, 0.5, 72.5);
   TH2F* hlnoise1 = new TH2F ("hlnoise1","hlnoise1",28, -14, 14, 72, 0.5, 72.5);
   TH2F* hlnoise2 = new TH2F ("hlnoise2","hlnoise2",28, -14, 14, 72, 0.5, 72.5);
   TH2F* hlnoise3 = new TH2F ("hlnoise3","hlnoise3",28, -14, 14, 72, 0.5, 72.5);
   TH2F* hlnoise4 = new TH2F ("hlnoise4","hlnoise4",28, -14, 14, 72, 0.5, 72.5);
   TH2F* hlnoise_stab4 = new TH2F ("hlnoise_stab4","hlnoise_stab4",1501, -0.5, 1500.5, 100, -0.1, 0.4);
   TH2F* hlnoise_stab3 = new TH2F ("hlnoise_stab3","hlnoise_stab3",1501, -0.5, 1500.5, 100, -0.1, 0.4);
   TH2F* hlnoise_stab2 = new TH2F ("hlnoise_stab2","hlnoise_stab2",1501, -0.5, 1500.5, 100, -0.1, 0.4);
   TH2F* hlnoise_stab1 = new TH2F ("hlnoise_stab1","hlnoise_stab1",1501, -0.5, 1500.5, 100, -0.1, 0.4);
   TH2F* hlnoise_stab6 = new TH2F ("hlnoise_stab6","hlnoise_stab6",1501, -0.5, 1500.5, 100, -0.1, 0.4);
   
   TH2F* hlnoise_stab_HF_d1 = new TH2F ("hlnoise_stab_HF_d1","hlnoise_stab_HF_d1",1501, -0.5, 1500.5, 100, -0.1, 0.4);
   TH2F* hlnoise_stab_HF_d2 = new TH2F ("hlnoise_stab_HF_d2","hlnoise_stab_HF_d2",1501, -0.5, 1500.5, 100, -0.1, 0.4);
   
   
   
   TH1F* h1signal = new TH1F ("h1signal","h1signal",100, -0.1,0.4);
   TH2F* hlsignal = new TH2F ("hlsignal","hlsignal",28, -14, 14, 72, 0.5, 72.5);
   TH2F* hlsignal1 = new TH2F ("hlsignal1","hlsignal1",28, -14, 14, 72, 0.5, 72.5);
   TH2F* hlsignal2 = new TH2F ("hlsignal2","hlsignal2",28, -14, 14, 72, 0.5, 72.5);
   TH2F* hlsignal3 = new TH2F ("hlsignal3","hlsignal3",28, -14, 14, 72, 0.5, 72.5);
   TH2F* hlsignal4 = new TH2F ("hlsignal4","hlsignal4",28, -14, 14, 72, 0.5, 72.5);
   
   
   Float_t myfile = 0.;
   
   
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      
      if(ientry == 0) {cout<<" New file "<<myfile<<endl; myfile++;}
      
      if(ieta == 0) continue;
//      if( ieta<0) {
//      cout<<" Mysubd "<<mysubd<<" "<<ieta<<" "<<" "<<iphi<<" "<<depth<<" "<<" "<<mom0_MB<<endl;
//      cout<<" Mom "<<mom2_Diff<<" "<<mom2_MB<<" "<<mom2_Noise<<endl;
//      cout<<" Mom "<<mom1_Diff<<" "<<mom1_MB<<" "<<mom1_Noise<<endl;
//      }
      if(mom0_MB == 0) continue;
      Float_t mean_MB = mom1_MB/mom0_MB;
      Float_t mean_NS = mom1_Noise/mom0_MB;
            
      Float_t disp_MB = mom2_MB/mom0_MB - (mom1_MB/mom0_MB)*(mom1_MB/mom0_MB);
      Float_t disp_NS = mom2_Noise/mom0_MB - (mom1_Noise/mom0_MB)*(mom1_Noise/mom0_MB);
      Float_t disp_Diff = mom2_Diff/mom0_MB - (mom1_Diff/mom0_MB)*(mom1_Diff/mom0_MB);
      
      if(mysubd == 1)
      {
      if(abs(ieta)<15) {
          h1noise->Fill(disp_NS); 
          if(disp_NS>0.22 && disp_NS<0.26) hlnoise4->Fill((Float_t)ieta,(Float_t)iphi,1.);
          if(disp_NS>0.16 && disp_NS<0.18) hlnoise3->Fill((Float_t)ieta,(Float_t)iphi,1.);
          if(disp_NS>0.12 && disp_NS<0.14) hlnoise2->Fill((Float_t)ieta,(Float_t)iphi,1.);
          if(disp_NS>0.11 && disp_NS<0.12) hlnoise1->Fill((Float_t)ieta,(Float_t)iphi,1.);
	  if(disp_NS>0.11) hlnoise->Fill((Float_t)ieta,(Float_t)iphi,disp_NS);
	  
          h1signal->Fill(disp_MB); 
          if(disp_NS>0.22 && disp_NS<0.26) {hlsignal4->Fill((Float_t)ieta,(Float_t)iphi,1.);}
          if(disp_NS>0.16 && disp_NS<0.18) {hlsignal3->Fill((Float_t)ieta,(Float_t)iphi,1.);}
          if(disp_NS>0.12 && disp_NS<0.14) {hlsignal2->Fill((Float_t)ieta,(Float_t)iphi,1.);}
          if(disp_NS>0.11 && disp_NS<0.12) {hlsignal1->Fill((Float_t)ieta,(Float_t)iphi,1.);}
	  if(disp_NS>0.11) hlsignal->Fill((Float_t)ieta,(Float_t)iphi,disp_MB);
	  
      }
      if( ieta == -1 && iphi == 70 ) {h1noise_mean_1->Fill(mean_NS);hlnoise_stab1->Fill(myfile,disp_NS,1.);}	
      if( ieta == -2 && iphi == 70 ) {h1noise_mean_2->Fill(mean_NS);hlnoise_stab2->Fill(myfile,disp_NS,1.);}	
      if( ieta == -3 && iphi == 70 ) {h1noise_mean_3->Fill(mean_NS);hlnoise_stab3->Fill(myfile,disp_NS,1.);}	
      if( ieta == -4 && iphi == 70 ) {h1noise_mean_4->Fill(mean_NS);hlnoise_stab4->Fill(myfile,disp_NS,1.);}	
      if( ieta == -6 && iphi == 70 ) {h1noise_mean_6->Fill(mean_NS);hlnoise_stab6->Fill(myfile,disp_NS,1.);}	
      }
      
      if( mysubd == 4 ){
       if(depth == 1) hlnoise_stab_HF_d1->Fill(myfile,disp_NS,1.);
       if(depth == 2) hlnoise_stab_HF_d2->Fill(myfile,disp_NS,1.);	
      }	
      
       if(ieta<0) {
// Calculation of dispersion ===============================
         
         mysubdetmin0[abs(ieta)][iphi][depth][mysubd] = mysubd;
	 
	 minmean_MB[abs(ieta)][iphi][depth][mysubd]=minmean_MB[abs(ieta)][iphi][depth][mysubd] + (float)mean_MB;
	 minmean_NS[abs(ieta)][iphi][depth][mysubd]=minmean_NS[abs(ieta)][iphi][depth][mysubd] + (float)mean_NS;
	 minmean_Diff[abs(ieta)][iphi][depth][mysubd]=minmean_Diff[abs(ieta)][iphi][depth][mysubd] + (float)disp_Diff;
	 minnevetaphi[abs(ieta)][iphi][depth][mysubd]++;
	 
         minerr_MB[abs(ieta)][iphi][depth][mysubd] = minerr_MB[abs(ieta)][iphi][depth][mysubd] + mean_MB*mean_MB;
         minerr_NS[abs(ieta)][iphi][depth][mysubd] = minerr_NS[abs(ieta)][iphi][depth][mysubd] + mean_NS*mean_NS;
         minerr_Diff[abs(ieta)][iphi][depth][mysubd] = minerr_Diff[abs(ieta)][iphi][depth][mysubd] + disp_Diff*disp_Diff;
	 
// ==========================================================
	
      }	
      if(ieta>=0) {

// Calculation of dispersion ===============================
         mysubdetpl0[abs(ieta)][iphi][depth][mysubd] = mysubd;
	 plmean_MB[abs(ieta)][iphi][depth][mysubd]=plmean_MB[abs(ieta)][iphi][depth][mysubd] + (float)mean_MB;
	 plmean_NS[abs(ieta)][iphi][depth][mysubd]=plmean_NS[abs(ieta)][iphi][depth][mysubd] + (float)mean_NS;
	 plmean_Diff[abs(ieta)][iphi][depth][mysubd]=plmean_Diff[abs(ieta)][iphi][depth][mysubd] + (float)disp_Diff;
	 plnevetaphi[abs(ieta)][iphi][depth][mysubd]++;
	 
         plerr_MB[abs(ieta)][iphi][depth][mysubd] = plerr_MB[abs(ieta)][iphi][depth][mysubd] + mean_MB*mean_MB;
         plerr_NS[abs(ieta)][iphi][depth][mysubd] = plerr_NS[abs(ieta)][iphi][depth][mysubd] + mean_NS*mean_NS;
         plerr_Diff[abs(ieta)][iphi][depth][mysubd] = plerr_Diff[abs(ieta)][iphi][depth][mysubd] + disp_Diff*disp_Diff;

	  
// ==========================================================

	 
      }
     
            
   } // jentry

   cout<<" Finish cycle "<<endl;
  
  Double_t plmeang_mean_MB = 0.;
  Double_t plmeang_mean_NS = 0.;
  Double_t plmeang_mean_Diff = 0.;
  Double_t plmeang_mean_Diff_av = 0.;
  
  Double_t minmeang_mean_MB = 0.;
  Double_t minmeang_mean_NS = 0.;
  Double_t minmeang_mean_Diff = 0.;
  Double_t minmeang_mean_Diff_av = 0.;
  
// ieta>0  
  for (int idep = 1; idep <5; idep++ )
  {
  for(int ietak = 1; ietak < 43; ietak++ )
  {
  for(int isub = 1; isub < 5; isub++ )
  {
    for(Int_t iphik = 1; iphik < 73; iphik++ )
    {
         if(mysubdetpl0[ietak][iphik][idep][isub] == 0) continue;
          if(plnevetaphi[ietak][iphik][idep][isub] == 0) {
            Float_t rr = 1.;
	    Float_t ss = 0.;
	    fprintf(Out1,"%d %d %d %d %.8f\n",mysubdetpl0[ietak][iphik][idep][isub],ietak,iphik,idep,ss);
	    fprintf(Out2,"%d %d %d %d %.8f\n",mysubdetpl0[ietak][iphik][idep][isub],ietak,iphik,idep,ss);
//	    fprintf(Out3,"%d %d %d %d %.8f\n",mysubdetpl0[ietak][iphik][idep][isub],ietak,iphik,idep,ss);
//	    fprintf(Out4,"%d %d %d %d %.8f\n",mysubdetpl0[ietak][iphik][idep][isub],ietak,iphik,idep,ss);
            continue;
           }
	           
// Mean dispersion
       plmeang_mean_MB = plmean_MB[ietak][iphik][idep][isub]/plnevetaphi[ietak][iphik][idep][isub];
       plmeang_mean_NS = plmean_NS[ietak][iphik][idep][isub]/plnevetaphi[ietak][iphik][idep][isub];
       plmeang_mean_Diff = plmean_Diff[ietak][iphik][idep][isub]/plnevetaphi[ietak][iphik][idep][isub];
       plmeang_mean_Diff_av = plmeang_mean_MB-plmeang_mean_NS;
      	 
       fprintf(Out1,"%d %d %d %d %.8f\n",mysubdetpl0[ietak][iphik][idep][isub],ietak,iphik,idep,plmeang_mean_MB);
       fprintf(Out2,"%d %d %d %d %.8f\n",mysubdetpl0[ietak][iphik][idep][isub],ietak,iphik,idep,plmeang_mean_NS);
//       fprintf(Out3,"%d %d %d %d %.8f\n",mysubdetpl0[ietak][iphik][idep][isub],ietak,iphik,idep,plmeang_mean_Diff);
//       fprintf(Out4,"%d %d %d %d %.8f\n",mysubdetpl0[ietak][iphik][idep][isub],ietak,iphik,idep,plmeang_mean_Diff_av);

    }
  }  
  }
  }
// ieta<0   
  for (int idep = 1; idep <5; idep++ )
  {
  for(int ietak = 1; ietak < 43; ietak++ )
  {
    Int_t iietak = -1.*ietak;
  for(int isub = 1; isub < 5; isub++ )
  {
    for(Int_t iphik = 1; iphik < 73; iphik++ )
    {
        if(mysubdetmin0[ietak][iphik][idep][isub] == 0) continue;
          if(minnevetaphi[ietak][iphik][idep][isub] == 0) {
            Float_t rr = 1.;
	    Float_t ss = 0.;
	    fprintf(Out1,"%d %d %d %d %.8f\n",mysubdetmin0[ietak][iphik][idep][isub],iietak,iphik,idep,ss);
	    fprintf(Out2,"%d %d %d %d %.8f\n",mysubdetmin0[ietak][iphik][idep][isub],iietak,iphik,idep,ss);
//	    fprintf(Out3,"%d %d %d %d %.8f\n",mysubdetmin0[ietak][iphik][idep][isub],iietak,iphik,idep,ss);
//	    fprintf(Out4,"%d %d %d %d %.8f\n",mysubdetmin0[ietak][iphik][idep][isub],iietak,iphik,idep,ss);
            continue;
           }
	           
// Mean dispersion
       minmeang_mean_MB = minmean_MB[ietak][iphik][idep][isub]/minnevetaphi[ietak][iphik][idep][isub];
       minmeang_mean_NS = minmean_NS[ietak][iphik][idep][isub]/minnevetaphi[ietak][iphik][idep][isub];
       minmeang_mean_Diff = minmean_Diff[ietak][iphik][idep][isub]/minnevetaphi[ietak][iphik][idep][isub];
      	 
       fprintf(Out1,"%d %d %d %d %.8f\n",mysubdetmin0[ietak][iphik][idep][isub],iietak,iphik,idep,minmeang_mean_MB);
       fprintf(Out2,"%d %d %d %d %.8f\n",mysubdetmin0[ietak][iphik][idep][isub],iietak,iphik,idep,minmeang_mean_NS);
//       fprintf(Out3,"%d %d %d %d %.8f\n",mysubdetmin0[ietak][iphik][idep][isub],iietak,iphik,idep,minmeang_mean_Diff);
//       fprintf(Out4,"%d %d %d %d %.8f\n",mysubdetmin0[ietak][iphik][idep][isub],iietak,iphik,idep,minmeang_mean_Diff_av);

    }
  }  
  }
  }
 
  TFile f("signal_noise_mean_8.9mln.root","recreate");
  h1noise->Write();
  hlnoise->Write();
  hlnoise1->Write();
  hlnoise2->Write();
  hlnoise3->Write();
  hlnoise4->Write();
  
  h1signal->Write();
  hlsignal->Write();
  hlsignal1->Write();
  hlsignal2->Write();
  hlsignal3->Write();
  hlsignal4->Write();
  
  h1noise_mean_1->Write();	
  h1noise_mean_2->Write();	
  h1noise_mean_3->Write();	
  h1noise_mean_4->Write();	
  h1noise_mean_6->Write();	
  hlnoise_stab4->Write();
  hlnoise_stab3->Write();
  hlnoise_stab2->Write();
  hlnoise_stab1->Write();
  hlnoise_stab6->Write();
  hlnoise_stab_HF_d1->Write();
  hlnoise_stab_HF_d2->Write();	
   
  fclose(Out1);
  fclose(Out2);
//  fclose(Out3);
//  fclose(Out4);         
}
