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
	      TChain *ch1,
	      TChain *ch2,
	      TString var = "ohMuL3Eta", TString cut = "", Int_t nbin = 100, Int_t min = -5.0, Int_t max = 5.0,
	      Double_t ksthreshold = 0.001)
{
  gROOT->SetStyle("Plain");

  cout << ch1->GetEntries() << endl
       << ch2->GetEntries() << endl;

  TH1F *h1 = new TH1F("h1","h1",nbin,min,max);
  TH1F *h2 = new TH1F("h2","h2",nbin,min,max);

  h1->SetMarkerStyle(20); h1->SetLineWidth(3);
  h2->SetMarkerStyle(20); h2->SetMarkerColor(2); h2->SetLineWidth(3); h2->SetLineColor(2);
  h1->Sumw2(); h2->Sumw2();

  ch1->Draw(var + " >> h1",cut,"e");
  ch2->Draw(var + " >> h2",cut,"e");

  // Scale to equal aread
  //  h1->Scale(1.0 * h2->GetEntries()/h1->GetEntries()); 

  // Scale to number of produced events per sample
  h1->Scale(1.0 / ch1->GetEntries());
  h2->Scale(1.0 / ch2->GetEntries());

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

void ValidateOpenHLTNtuples(TString d1 = "rfio:/castor/cern.ch/user/f/fwyzard/OpenHLT/MinBias/lumi8e29/Summer08_MinBias_hltanalyzer_redoL1_StartupV8_L1StartupMenu_5.root",
			    TString d2 = "rfio:/castor/cern.ch/user/a/apana/Summer09_312/MinBias/8e29/Summer08_MinBias_hltanalyzer_Startup31X_V2_5.root",
			    Double_t threshold = 1000.001)
{
  TFile *f = new TFile("openhltvalidation.root","recreate");

  TChain *dir1 = new TChain("HltTree");
  TChain *dir2 = new TChain("HltTree");

  dir1->Add(d1);
  dir2->Add(d2);

  //  dir1->Add("rfio:/castor/cern.ch/user/f/fwyzard/OpenHLT/MinBias/lumi8e29/Summer08_MinBias_hltanalyzer_redoL1_StartupV8_L1StartupMenu_6*.root");
  //  dir1->Add("rfio:/castor/cern.ch/user/f/fwyzard/OpenHLT/MinBias/lumi8e29/Summer08_MinBias_hltanalyzer_redoL1_StartupV8_L1StartupMenu_7*.root");

  //  dir2->Add("rfio:/castor/cern.ch/user/a/apana/Summer09_312/MinBias/8e29/Summer08_MinBias_hltanalyzer_Startup31X_V2_5.root");
  //  dir2->Add("rfio:/castor/cern.ch/user/a/apana/Summer09_312/MinBias/8e29/Summer08_MinBias_hltanalyzer_Startup31X_V2_6.root");
  //  dir2->Add("rfio:/castor/cern.ch/user/a/apana/Summer09_312/MinBias/8e29/Summer08_MinBias_hltanalyzer_Startup31X_V2_7.root");



  DoKSTest(f, dir1, dir2, "ohEleEtaLW[0]", "ohElePixelSeedsLW[0] > 0", 100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "ohElePhiLW[0]", "ohElePixelSeedsLW[0] > 0", 100, -4, 4, threshold);
  DoKSTest(f, dir1, dir2, "ohEleEtLW[0]", "ohElePixelSeedsLW[0] > 0", 100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "ohElePixelSeedsLW[0]", "", 10, 0, 10, threshold);
  DoKSTest(f, dir1, dir2, "ohEleHisoLW[0]", "ohElePixelSeedsLW[0] > 0", 100, 0, 20, threshold);
  DoKSTest(f, dir1, dir2, "ohEleTisoLW[0]", "ohElePixelSeedsLW[0] > 0 && ohEleTisoLW[0] > -1", 100, 0, 20, threshold);

  DoKSTest(f, dir1, dir2, "L1MuEta[0]", "L1MuPt[0] > -1", 100, -5, 5, threshold);   
  DoKSTest(f, dir1, dir2, "L1MuPhi[0]", "L1MuPt[0] > -1", 100, -4, 4, threshold);    
  DoKSTest(f, dir1, dir2, "L1MuPt[0]", "L1MuPt[0] > -1", 100, 0, 100, threshold);    
  DoKSTest(f, dir1, dir2, "ohMuL2Eta[0]", "", 100, -5, 5, threshold);  
  DoKSTest(f, dir1, dir2, "ohMuL2Phi[0]", "", 100, -4, 4, threshold);   
  DoKSTest(f, dir1, dir2, "ohMuL2Pt[0]", "", 100, 0, 100, threshold);   
  DoKSTest(f, dir1, dir2, "ohMuL3Eta[0]", "", 100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "ohMuL3Phi[0]", "", 100, -4, 4, threshold); 
  DoKSTest(f, dir1, dir2, "ohMuL3Pt[0]", "", 100, 0, 100, threshold); 
  DoKSTest(f, dir1, dir2, "ohMuL3Iso[0]", "", 2, 0, 2, threshold);

  DoKSTest(f, dir1, dir2, "L1IsolEmEta[0]", "L1IsolEmEt[0] > -1", 100, -5, 5, threshold);    
  DoKSTest(f, dir1, dir2, "L1IsolEmPhi[0]", "L1IsolEmEt[0] > -1", 100, -4, 4, threshold);     
  DoKSTest(f, dir1, dir2, "L1IsolEmEt[0]", "L1IsolEmEt[0] > -1", 100, 0, 100, threshold);     
  DoKSTest(f, dir1, dir2, "L1NIsolEmEta[0]", "L1NIsolEmEt[0] > -1", 100, -5, 5, threshold);     
  DoKSTest(f, dir1, dir2, "L1NIsolEmPhi[0]", "L1NIsolEmEt[0] > -1", 100, -4, 4, threshold);      
  DoKSTest(f, dir1, dir2, "L1NIsolEmEt[0]", "L1NIsolEmEt[0] > -1", 100, 0, 100, threshold); 

  DoKSTest(f, dir1, dir2, "ohPhotEta[0]", "", 100, -5, 5, threshold);  
  DoKSTest(f, dir1, dir2, "ohPhotPhi[0]", "", 100, -4, 4, threshold);   
  DoKSTest(f, dir1, dir2, "ohPhotEt[0]", "", 100, 0, 100, threshold);   
  DoKSTest(f, dir1, dir2, "ohPhotHiso[0]", "", 100, 0, 20, threshold);
  DoKSTest(f, dir1, dir2, "ohPhotEiso[0]", "", 100, 0, 20, threshold);

  DoKSTest(f, dir1, dir2, "L1HfRing1EtSumNegativeEta", "", 20, 0, 20, threshold);   
  DoKSTest(f, dir1, dir2, "L1HfRing1EtSumPositiveEta", "", 20, 0, 20, threshold);    
  DoKSTest(f, dir1, dir2, "L1HfRing2EtSumNegativeEta", "", 20, 0, 20, threshold);    
  DoKSTest(f, dir1, dir2, "L1HfRing2EtSumPositiveEta", "", 20, 0, 20, threshold);    
  DoKSTest(f, dir1, dir2, "L1HfTowerCountNegativeEtaRing1", "", 20, 0, 20, threshold);     
  DoKSTest(f, dir1, dir2, "L1HfTowerCountPositiveEtaRing1", "", 20, 0, 20, threshold);      
  DoKSTest(f, dir1, dir2, "L1HfTowerCountNegativeEtaRing2", "", 20, 0, 20, threshold);      
  DoKSTest(f, dir1, dir2, "L1HfTowerCountPositiveEtaRing2", "", 20, 0, 20, threshold);      

  DoKSTest(f, dir1, dir2, "L1CenJetEta[0]", "L1CenJetEt[0] > -1", 100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "L1CenJetPhi[0]", "L1CenJetEt[0] > -1", 100, -4, 4, threshold);
  DoKSTest(f, dir1, dir2, "L1CenJetEt[0]", "L1CenJetEt[0] > -1", 100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "L1ForJetEta[0]", "L1ForJetEt[0] > -1", 100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "L1ForJetPhi[0]", "L1ForJetEt[0] > -1", 100, -4, 4, threshold);
  DoKSTest(f, dir1, dir2, "L1ForJetEt[0]", "L1ForJetEt[0] > -1", 100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "recoJetCorCalEta[0]", "", 100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "recoJetCorCalPhi[0]", "", 100, -4, 4, threshold);
  DoKSTest(f, dir1, dir2, "recoJetCorCalPt[0]", "", 100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "recoMetCal", "", 100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "recoMetCalSum", "", 100, 0, 100, threshold);

  DoKSTest(f, dir1, dir2, "ohTauPhi[0]","",100, -4, 4, threshold);
  DoKSTest(f, dir1, dir2, "ohTauEta[0]","",100, -5, 5, threshold);
  DoKSTest(f, dir1, dir2, "ohTauPt[0]","",100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "ohTauL25Tpt[0]","",100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "ohTauL3Tpt[0]","",100, 0, 100, threshold);
  DoKSTest(f, dir1, dir2, "ohTauL25Tiso[0]","",2, 0, 2, threshold);
  DoKSTest(f, dir1, dir2, "ohTauL3Tiso[0]","",2, 0, 2, threshold);

  f->Close();
}
