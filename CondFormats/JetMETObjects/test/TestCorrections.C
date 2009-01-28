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
  TFile  *file = new TFile("/uscms_data/d2/kkousour/0430FBCD-7FCB-DD11-AA56-0019B9E50135.root");
  TTree *tree = (TTree*)file->Get("Events");
  vector<reco::CaloJet> CaloJetCollection;
  TBranch *branch = tree->GetBranch(tree->GetAlias("IC5CaloJet"));
  branch->SetAddress(&CaloJetCollection); 

  ////////////// Defining the vector of correction levels ////////
  vector<string> Levels;
  Levels.push_back("L2");
  Levels.push_back("L3");
  Levels.push_back("L4");
  Levels.push_back("L5");
  Levels.push_back("L7");
  ////////////// Defining the vector of data filename tags ///////
  vector<string> CorrectionTags;
  CorrectionTags.push_back("Summer08_L2Relative_IC5Calo");
  CorrectionTags.push_back("Summer08_L3Absolute_IC5Calo");
  CorrectionTags.push_back("CMSSW_152_L4EMF");
  CorrectionTags.push_back("L5Flavor_fromQCD_iterativeCone5");
  CorrectionTags.push_back("L7parton_IC5_080921");
  ////////////// Defining the flavor and parton options //////////
  string FlavorOption("uds");
  string PartonOption("jJ");
  ////////////// Defining the JetCorrector ///////////////////////
  FWLiteJetCorrector JetCorrector(Levels,CorrectionTags,FlavorOption,PartonOption);
  
  ////////////// Fill demo histogram  ////////////////////////////
  TH2F *hMapping = new TH2F("Mapping","Mapping",500,0,500,500,0,500);
  for(unsigned int index = 0; index < 200; ++index ) 
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
	  double scale = JetCorrector.getCorrection(pt,eta,emf);
	  vector<double> factors = JetCorrector.getSubCorrections(pt,eta,emf);
	  cout<<"Pt = "<<pt<<", Eta = "<<eta<<", EMF = "<<emf<<", correction = "<<scale<<", CorPt = "<<scale*pt<<endl;
          hMapping->Fill(pt,scale*pt);
	  for(unsigned int i=0;i<factors.size();i++)
	    cout<<Levels[i]<<" = "<<factors[i]<<endl;
        }
    }
  ////////////// Draw demo histogram /////////////////////////////
  TCanvas *c = new TCanvas("CorrectionMapping","CorrectionMapping");
  hMapping->SetTitle("Correction Mapping");
  hMapping->GetXaxis()->SetTitle("Uncorrected jet pT (GeV)");
  TString tmp("");
  for(unsigned int i=0;i<Levels.size();i++)
    tmp+=Levels[i];
  tmp+=" Corrected jet pT (GeV)";
  char title[1024];
  sprintf(title,"%s",tmp);
  hMapping->GetYaxis()->SetTitle(tmp);
  hMapping->Draw();
}
