void TestCorrections()
{
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1);
  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable(); 
  using namespace reco;
  using namespace edm; 
  using namespace std;
  main();
}  
////////////////////////////////////////////////
void main()
{
  ////////////// Getting the jet collection //////////////////////
  TFile  *file = new TFile("/uscms_data/d2/kkousour/PAT/CMSSW_2_2_6/src/ttbar_pat_v2_numEvent10.root");
  TTree *tree = (TTree*)file->Get("Events");
  vector<reco::CaloJet> CaloJetCollection;
  TBranch *branch = tree->GetBranch(tree->GetAlias("IC5CaloJet"));
  branch->SetAddress(&CaloJetCollection); 
  ////////////// Defining the L2L3L5L7JetCorrector ///////////////////////
  string Levels = "L2:L3:L5:L7";
  string Tags = "Summer08Redigi_L2Relative_IC5Calo:Summer08Redigi_L3Absolute_IC5Calo:L5Flavor_IC5:L7Parton_IC5";
  string Options = "Flavor:gJ & Parton:gJ";
  CombinedJetCorrector *L2L3L5L7JetCorrector = new CombinedJetCorrector(Levels,Tags,Options);
  ////////////// Defining the L2L3JetCorrector ///////////////////////
  string Levels = "L2:L3";
  string Tags = "Summer08Redigi_L2Relative_IC5Calo:Summer08Redigi_L3Absolute_IC5Calo";
  CombinedJetCorrector *L2L3JetCorrector = new CombinedJetCorrector(Levels,Tags);
  ////////////// Fill demo histogram  ////////////////////////////
  TH2F *hMapping = new TH2F("Mapping","Mapping",500,0,500,500,0,500);
  TH2F *hRatio = new TH2F("Ratio","Ratio",500,0,500,500,0.8,1.2);
  for(unsigned int index = 0; index < 10; ++index ) 
    {
      cout << "Entry index: " << index << endl;  
      branch->GetEntry(index);
      int numJets = CaloJetCollection.size();
      for(unsigned int jetIndex = 0; jetIndex < 2; ++jetIndex ) 
        {
	  CaloJet* Jet = (CaloJet*)CaloJetCollection[jetIndex];
          double pt = Jet->pt();    
          double eta = Jet->eta();  
          double emf = Jet->emEnergyFraction();  
	  double L2L3scale = L2L3JetCorrector->getCorrection(pt,eta,emf);
          double L2L3L5L7scale = L2L3L5L7JetCorrector->getCorrection(pt,eta,emf);
          vector<double> L2L3factors = L2L3JetCorrector->getSubCorrections(pt,eta,emf);
	  vector<double> L2L3L5L7factors = L2L3L5L7JetCorrector->getSubCorrections(pt,eta,emf);
	  cout<<"Pt = "<<pt<<", Eta = "<<eta<<", EMF = "<<emf<<endl;
          cout<<"L2L3correction = "<<L2L3scale<<", L2L3CorPt = "<<L2L3scale*pt<<endl;
          for(unsigned int i=0;i<L2L3factors.size();i++)
	    cout<<L2L3factors[i]<<endl;
          cout<<"L2L3L5L7correction = "<<L2L3L5L7scale<<", L2L3L5L7CorPt = "<<L2L3L5L7scale*pt<<endl;
          for(unsigned int i=0;i<L2L3L5L7factors.size();i++)
	    cout<<L2L3L5L7factors[i]<<endl;
          hMapping->Fill(L2L3scale*pt,L2L3L5L7scale*pt);
          hRatio->Fill(L2L3scale*pt,L2L3L5L7scale/L2L3scale);	  
        }
    }
  ////////////// Draw demo histograms /////////////////////////////
  TCanvas *c = new TCanvas("CorrectionMapping","CorrectionMapping");
  hMapping->SetTitle("Correction Mapping");
  hMapping->GetXaxis()->SetTitle("L2L3Corrected jet pT (GeV)");
  hMapping->GetYaxis()->SetTitle("L2L3L5L7Corrected jet pT (GeV)");
  hMapping->Draw();

  TCanvas *c = new TCanvas("RatioMapping","RatioMapping");
  hRatio->SetTitle("Ratio Mapping");
  hRatio->GetXaxis()->SetTitle("L2L3Corrected jet pT (GeV)");
  hRatio->GetYaxis()->SetTitle("L2L3L5L7/L2L3");
  hRatio->Draw(); 
}
