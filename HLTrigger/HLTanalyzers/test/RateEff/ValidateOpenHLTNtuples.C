/*
 * ValidateOpenHLTNtuples.C
 *
 * Macro for comparing distributions between any 2 versions of OpenHLT ntuples.
 *
 * Usage: ValidateOpenHLTNtuples(TString sample1, TString sample2,  Double_t threshold)
 * 
 * Where sample1 and sample2 are the two sets of ntuples to compare, and threshold is the 
 * minimum level of compatibility (based on a KS-test) allowed between the two histograms.
 * The list of variables to compare is defined in void ValidateOpenHLTNtuples.
 *
 */

void DoKSTest(TFile *histfile, 
	      TString sample1 = "/uscmst1b_scratch/lpc1/lpctrig/apana/data/MinBias/lumi8e29/*", 
	      TString sample2 = "/uscmst1b_scratch/lpc1/lpctrig/apana/data/MinBias/lumi1e31_newL1/*",
	      TString var = "ohMuL3Eta", TString cut = "", Int_t nbin = 100, Int_t min = -5.0, Int_t max = 5.0,
	      Double_t ksthreshold = 0.001)
{
  gROOT->SetStyle("Plain");

  TChain *ch1 = new TChain("HltTree");
  TChain *ch2 = new TChain("HltTree");

  ch1->Add(sample1);
  ch2->Add(sample2);

  TH1F *h1 = new TH1F("h1","h1",nbin,min,max);
  TH1F *h2 = new TH1F("h2","h2",nbin,min,max);

  h1->SetMarkerStyle(20); h1->SetLineWidth(3);
  h2->SetMarkerStyle(20); h2->SetMarkerColor(2); h2->SetLineWidth(3); h2->SetLineColor(2);
  h1->Sumw2(); h2->Sumw2();

  ch1->Draw(var + " >> h1",cut,"e");
  ch2->Draw(var + " >> h2",cut,"e");

  h1->Scale(1.0 * h2->GetEntries()/h1->GetEntries()); 
  if(h1->GetMaximum() > h2->GetMaximum())
    h1->SetMaximum(1.5 * h1->GetMaximum());
  else
    h1->SetMaximum(1.5 * h2->GetMaximum());

  TCanvas *c1 = new TCanvas(var,var);

  h1->SetXTitle(var); 

  h1->Draw("e");
  h2->Draw("esame");

  Double_t ksprob = h1->KolmogorovTest(h2);
  TString htitle = "KS probability = ";
  htitle += ksprob;
  h1->SetTitle(htitle);

  if(ksprob < ksthreshold)
    {
      histfile->cd();
      c1->Write();
      cout << "P(" << var << ") = " << ksprob << " ---- FAILED COMPARISON!" << endl;  
    }
  else
    cout << "P(" << var << ") = " << ksprob << endl; 
  
  delete c1;
  delete h1;
  delete h2;
}

void ValidateOpenHLTNtuples(TString dir1 = "/uscmst1b_scratch/lpc1/lpctrig/apana/data/MinBias/lumi8e29/Summer08_MinBias_hltanalyzer_redoL1_StartupV8_L1StartupMenu_8*", 
			    TString dir2 = "/uscmst1b_scratch/lpc1/lpctrig/apana/data/MinBias/lumi1e31_newL1/Summer08_MinBias_hltanalyzer_redoL1_StartupV8_L1DefaultMenu_*",
			    Double_t threshold = 0.001)
{
  TFile *f = new TFile("openhltvalidation.root","recreate");

  DoKSTest(f, dir1, dir2, "L1MuEta", "L1MuPt > -1", 100, -5, 5, threshold);   
  DoKSTest(f, dir1, dir2, "L1MuPhi", "L1MuPt > -1", 100, -4, 4, threshold);    
  DoKSTest(f, dir1, dir2, "L1MuPt", "L1MuPt > -1", 100, 0, 100, threshold);    
  DoKSTest(f, dir1, dir2, "ohMuL2Eta", "", 100, -5, 5, threshold);  
  DoKSTest(f, dir1, dir2, "ohMuL2Phi", "", 100, -4, 4, threshold);   
  DoKSTest(f, dir1, dir2, "ohMuL2Pt", "", 100, 0, 100, threshold);   
  DoKSTest(f, dir1, dir2, "ohMuL3Eta", "", 100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "ohMuL3Phi", "", 100, -4, 4, threshold); 
  DoKSTest(f, dir1, dir2, "ohMuL3Pt", "", 100, 0, 100, threshold); 
  DoKSTest(f, dir1, dir2, "ohMuL3Iso", "", 2, 0, 2, threshold);

  DoKSTest(f, dir1, dir2, "L1IsolEmEta", "L1IsolEmEt > -1", 100, -5, 5, threshold);    
  DoKSTest(f, dir1, dir2, "L1IsolEmPhi", "L1IsolEmEt > -1", 100, -4, 4, threshold);     
  DoKSTest(f, dir1, dir2, "L1IsolEmEt", "L1IsolEmEt > -1", 100, 0, 100, threshold);     
  DoKSTest(f, dir1, dir2, "L1NIsolEmEta", "L1NIsolEmEt > -1", 100, -5, 5, threshold);     
  DoKSTest(f, dir1, dir2, "L1NIsolEmPhi", "L1NIsolEmEt > -1", 100, -4, 4, threshold);      
  DoKSTest(f, dir1, dir2, "L1NIsolEmEt", "L1NIsolEmEt > -1", 100, 0, 100, threshold);      
  DoKSTest(f, dir1, dir2, "ohEleEta", "", 100, -5, 5, threshold); 
  DoKSTest(f, dir1, dir2, "ohElePhi", "", 100, -4, 4, threshold);  
  DoKSTest(f, dir1, dir2, "ohEleEt", "", 100, 0, 100, threshold);  
  DoKSTest(f, dir1, dir2, "ohElePixelSeeds", "", 10, 0, 10, threshold);
  DoKSTest(f, dir1, dir2, "ohEleHiso", "", 100, 0, 20, threshold);
  DoKSTest(f, dir1, dir2, "ohEleTiso", "ohEleTiso > -1", 100, 0, 20, threshold);

  DoKSTest(f, dir1, dir2, "ohPhotEta", "", 100, -5, 5, threshold);  
  DoKSTest(f, dir1, dir2, "ohPhotPhi", "", 100, -4, 4, threshold);   
  DoKSTest(f, dir1, dir2, "ohPhotEt", "", 100, 0, 100, threshold);   
  DoKSTest(f, dir1, dir2, "ohPhotHiso", "", 100, 0, 20, threshold);
  DoKSTest(f, dir1, dir2, "ohPhotEiso", "", 100, 0, 20, threshold);

  DoKSTest(f, dir1, dir2, "L1HfRing1EtSumNegativeEta", "", 20, 0, 20, threshold);   
  DoKSTest(f, dir1, dir2, "L1HfRing1EtSumPositiveEta", "", 20, 0, 20, threshold);    
  DoKSTest(f, dir1, dir2, "L1HfRing2EtSumNegativeEta", "", 20, 0, 20, threshold);    
  DoKSTest(f, dir1, dir2, "L1HfRing2EtSumPositiveEta", "", 20, 0, 20, threshold);    
  DoKSTest(f, dir1, dir2, "L1HfTowerCountNegativeEtaRing1", "", 20, 0, 20, threshold);     
  DoKSTest(f, dir1, dir2, "L1HfTowerCountPositiveEtaRing1", "", 20, 0, 20, threshold);      
  DoKSTest(f, dir1, dir2, "L1HfTowerCountNegativeEtaRing2", "", 20, 0, 20, threshold);      
  DoKSTest(f, dir1, dir2, "L1HfTowerCountPositiveEtaRing2", "", 20, 0, 20, threshold);      

  DoKSTest(f, dir1, dir2, "L1CenJetEta", "L1CenJetEt > -1", 100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "L1CenJetPhi", "L1CenJetEt > -1", 100, -4, 4, threshold);
  DoKSTest(f, dir1, dir2, "L1CenJetEt", "L1CenJetEt > -1", 100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "L1ForJetEta", "L1ForJetEt > -1", 100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "L1ForJetPhi", "L1ForJetEt > -1", 100, -4, 4, threshold);
  DoKSTest(f, dir1, dir2, "L1ForJetEt", "L1ForJetEt > -1", 100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "recoJetCorCalEta", "", 100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "recoJetCorCalPhi", "", 100, -4, 4, threshold);
  DoKSTest(f, dir1, dir2, "recoJetCorCalPt", "", 100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "recoMetCal", "", 100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "recoMetCalSum", "", 100, 0, 100, threshold);

  DoKSTest(f, dir1, dir2, "ohTauPhi","",100, -4, 4, threshold);
  DoKSTest(f, dir1, dir2, "ohTauEta","",100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "ohTauPt","",100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "ohTauL25Tpt","",100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "ohTauL3Tpt","",100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "ohTauL25Tiso","",2, 0, 2, threshold);
  DoKSTest(f, dir1, dir2, "ohTauL3Tiso","",2, 0, 2, threshold);

  f->Close();
}
