#include "DataFormats/Math/interface/deltaR.h"
#include "DQM/Physics/interface/MEzCalculator.h"
#include "DQM/Physics/interface/SemiLeptonChecker.h"

SemiLeptonChecker::SemiLeptonChecker(const edm::ParameterSet& iConfig, std::string relativePath, std::string label)
{
  //now do what ever initialization is needed
  dqmStore_ = edm::Service<DQMStore>().operator->();
  leptonType_       = iConfig.getParameter<std::string>( "leptonType" );
  NbOfEvents        = 0;
  relativePath_     = relativePath;
  label_            = label;

  found_goodMET_    = false;
  isMuon_           = true;
  if (leptonType_ == "electron") isMuon_ = false;
}


SemiLeptonChecker::~SemiLeptonChecker()
{
  delete dqmStore_;
}

void
SemiLeptonChecker::analyze(const std::vector<reco::CaloJet>& jets, bool useJES, const std::vector<reco::CaloMET>& mets, const std::vector<reco::Muon>& muons, const std::vector<reco::GsfElectron>& electrons, const edm::Event& iEvent, const edm::EventSetup& iSetup){
  //using namespace edm;
  NbOfEvents++;
  
  TLorentzVector METP4;
  TLorentzVector leptonP4;
  TLorentzVector nuP4;
  TLorentzVector lepWP4;
  TLorentzVector hadWP4;
  TLorentzVector topPairP4;
  TLorentzVector hadTopP4;
  TLorentzVector lepTopP4;
  
  // Histograms related to Muon/Electron and Jets
  std::vector< TLorentzVector > vectorjets;
  double cutNgoodJets = jets.size();
  if (cutNgoodJets > 6) cutNgoodJets = 6;
  double corrJES = 1.;
  for( size_t ijet=0; ijet != cutNgoodJets; ++ijet ) {
    TLorentzVector jetP4;
    reco::CaloJet correctedJet = jets[ijet];
    if(useJES){
      corrJES = acorrector->correction((jets)[ijet], iEvent, iSetup);
      correctedJet.scaleEnergy(corrJES);
    }
    jetP4.SetPxPyPzE(correctedJet.px(),correctedJet.py(),correctedJet.pz(),correctedJet.energy());
    vectorjets.push_back(jetP4);
  }
  
  METP4.SetPxPyPzE(mets[0].px(),mets[0].py(),mets[0].pz(),mets[0].energy());
  if (isMuon_)
    leptonP4.SetPxPyPzE(muons[0].px(),muons[0].py(),muons[0].pz(),muons[0].energy());
  else
    leptonP4.SetPxPyPzE(electrons[0].px(),electrons[0].py(),electrons[0].pz(),electrons[0].energy());
  
  double minDeltaR_lepton_jet = 9e9;
  for( size_t ijet=0; ijet < jets.size(); ++ijet ) {
    TLorentzVector tmpP4;
    tmpP4.SetPxPyPzE(jets[ijet].px(),jets[ijet].py(),jets[ijet].pz(),jets[ijet].energy());
    double aDeltaR_lepton_jet = deltaR(tmpP4.Eta(), tmpP4.Phi(), leptonP4.Eta(), leptonP4.Phi() );
    histocontainerC_["deltaR_jet_"+leptonType_]->Fill(aDeltaR_lepton_jet);
    if ( aDeltaR_lepton_jet < minDeltaR_lepton_jet ) {
      minDeltaR_lepton_jet = aDeltaR_lepton_jet;
    }
  }
  histocontainerC_["deltaR_jet_"+leptonType_+"_min"]->Fill(minDeltaR_lepton_jet);
  
  // Calculate neutrino Pz and Top reonstruction
  found_goodMET_    = false;
  if (mets.size() == 0) return;
  
  double neutrinoPz = -999999.;
  MEzCalculator zcalculator;
  zcalculator.SetMET( METP4 );
  zcalculator.SetLepton( leptonP4, isMuon_ );
  neutrinoPz = zcalculator.Calculate(1);// 1 = closest to the lepton Pz, 3 = largest cosineCM
  
  nuP4.SetPxPyPzE(METP4.Px(), METP4.Py(), neutrinoPz,
		  sqrt(METP4.Px()*METP4.Px()+METP4.Py()*METP4.Py()+neutrinoPz*neutrinoPz) );
  
  lepWP4 = leptonP4 + nuP4;
  
  histocontainerC_["LeptonicWMass"]->Fill(lepWP4.M());

  edm::LogInfo("Debug|SemiLeptonicChecker") << "[SemiLeptonChecker] Leptonic W mass = " << lepWP4.M() << std::endl;
  if (lepWP4.M() < 150.) {
    found_goodMET_ = true;
    edm::LogInfo("Debug|SemiLeptonicChecker") << "[SemiLeptonChecker] Found good MET" << std::endl;
  }
  
  myCombi0_.SetLeptonicW( lepWP4 );
  myCombi0_.FourJetsCombinations(vectorjets);
  
  Combo bestCombo = myCombi0_.GetCombinationSumEt(0);
  
  hadWP4 = bestCombo.GetHadW();
  hadTopP4 = bestCombo.GetHadTop();
  lepTopP4 = bestCombo.GetLepTop();
  
  //Fill M3 Histograms
  histocontainer_[0]["HadronicTopMass"]->Fill(hadTopP4.M());
  histocontainer_[0]["LeptonicTopMass"]->Fill(lepTopP4.M());
  histocontainer_[0]["HadronicWMass"]->Fill(hadWP4.M());
  histocontainer_[0]["HadronicTopPt"]->Fill(hadTopP4.Pt());
  histocontainer_[0]["LeptonicTopPt"]->Fill(lepTopP4.Pt());
  histocontainer_[0]["HadronicTopEta"]->Fill(hadTopP4.Eta());
  histocontainer_[0]["LeptonicTopEta"]->Fill(lepTopP4.Eta());
  
  // Fill M3prime Histograms
  if (found_goodMET_){
    myCombi1_.SetLeptonicW( lepWP4 );
    myCombi1_.UseMtopConstraint(true);
    myCombi1_.SetSigmas(0);
    myCombi1_.FourJetsCombinations(vectorjets);

    Combo bestCombo = myCombi1_.GetCombination(0);
    hadWP4 = bestCombo.GetHadW();
    hadTopP4 = bestCombo.GetHadTop();
    lepTopP4 = bestCombo.GetLepTop();
    
    //Fill Histograms
    histocontainer_[1]["HadronicTopMass"]->Fill(hadTopP4.M());
    histocontainer_[1]["LeptonicTopMass"]->Fill(lepTopP4.M());
    histocontainer_[1]["HadronicWMass"]->Fill(hadWP4.M());
    histocontainer_[1]["HadronicTopPt"]->Fill(hadTopP4.Pt());
    histocontainer_[1]["LeptonicTopPt"]->Fill(lepTopP4.Pt());
    histocontainer_[1]["HadronicTopEta"]->Fill(hadTopP4.Eta());
    histocontainer_[1]["LeptonicTopEta"]->Fill(lepTopP4.Eta());
  }
}

void 
SemiLeptonChecker::beginJob(const edm::EventSetup& iSetup, std::string jetCorrector)
{
  acorrector = JetCorrector::getJetCorrector(jetCorrector,iSetup);
  dqmStore_->setCurrentFolder(relativePath_+"/"+label_+"_Common");
  std::string suffix_ = std::string(isMuon_?"#mu":"e");
  histocontainerC_["LeptonicWMass"] = dqmStore_->book1D("LeptonicW_mass","Mass("+suffix_+" + #nu) [GeV/c^{2}]",30,0.,300.);
  histocontainerC_["LeptonicWMass"]->setAxisTitle("[GeV/c^{2}]",1);
  histocontainerC_["deltaR_jet_"+leptonType_] = dqmStore_->book1D("deltaR_jet_"+leptonType_,"#Delta R("+suffix_+", jet)",35,0,9);
  histocontainerC_["deltaR_jet_"+leptonType_]->setAxisTitle("#DeltaR("+suffix_+", jet)",1);
  histocontainerC_["deltaR_jet_"+leptonType_+"_min"] = dqmStore_->book1D("deltaR_jet_"+leptonType_+"_min","Min. #Delta R("+suffix_+", jet)",35,0,9);
  histocontainerC_["deltaR_jet_"+leptonType_+"_min"]->setAxisTitle("Min. #DeltaR("+suffix_+", jet)",1);

  dqmStore_->setCurrentFolder(relativePath_+"/"+label_+"_M3");
  histocontainer_[0]["HadronicTopMass"] = dqmStore_->book1D("HadronicTop_mass","Mass (j_{1}, j_{2}, j_{3}) [GeV/c^{2}]",50,0.,500.);
  histocontainer_[0]["HadronicTopMass"]->setAxisTitle("Hadronic Top mass (M3) [GeV/c^{2}]",1);
  histocontainer_[0]["LeptonicTopMass"] = dqmStore_->book1D("LeptonicTop_mass","Mass (j_{4}+W_{"+suffix_+"+#nu}) [GeV/c^{2}]",50,0.,500.);
  histocontainer_[0]["LeptonicTopMass"]->setAxisTitle("Leptonic Top mass [GeV/c^{2}]",1);
  histocontainer_[0]["HadronicWMass"] = dqmStore_->book1D("HadronicW_mass","Mass (j_{1}, j_{2}) [GeV/c^{2}]",30,0.,300.);
  histocontainer_[0]["HadronicWMass"]->setAxisTitle("Hadronic W mass [GeV/c^{2}]",1);
  histocontainer_[0]["HadronicTopPt"] = dqmStore_->book1D("HadronicTop_pT","Top p_{T}",30,0.,300.);
  histocontainer_[0]["HadronicTopPt"]->setAxisTitle("Hadronic Top p_{T} [GeV/c]",1);
  histocontainer_[0]["LeptonicTopPt"] = dqmStore_->book1D("LeptonicTop_pT","Top p_{T}",30,0.,300.);
  histocontainer_[0]["LeptonicTopPt"]->setAxisTitle("Leptonic Top p_{T} [GeV/c]",1);
  histocontainer_[0]["HadronicTopEta"] = dqmStore_->book1D("HadronicTop_eta","Top #eta",50,-5.,5.);
  histocontainer_[0]["HadronicTopEta"]->setAxisTitle("Hadronic Top #eta",1);
  histocontainer_[0]["LeptonicTopEta"] = dqmStore_->book1D("LeptonicTop_eta","Top #eta",50,-5.,5.);
  histocontainer_[0]["LeptonicTopEta"]->setAxisTitle("Leptonic Top #eta",1);
  
  dqmStore_->setCurrentFolder(relativePath_+"/"+label_+"_M3prime");
  histocontainer_[1]["HadronicTopMass"] = dqmStore_->book1D("HadronicTop_mass","Mass (j_{1}, j_{2}, j_{3}) [GeV/c^{2}]",50,0.,500.);
  histocontainer_[1]["HadronicTopMass"]->setAxisTitle("Hadronic Top mass (M3') [GeV/c^{2}]",1);
  histocontainer_[1]["LeptonicTopMass"] = dqmStore_->book1D("LeptonicTop_mass","Mass (j_{4}, W_{"+suffix_+" + #nu}) [GeV/c^{2}]",50,0.,500.);
  histocontainer_[1]["LeptonicTopMass"]->setAxisTitle("Leptonic Top mass [GeV/c^{2}]",1);
  histocontainer_[1]["HadronicWMass"] = dqmStore_->book1D("HadronicW_mass","Mass (j_{1}, j_{2}) [GeV/c^{2}]",30,0.,300.);
  histocontainer_[1]["HadronicWMass"]->setAxisTitle("Hadronic W mass [GeV/c^{2}]",1);
  histocontainer_[1]["HadronicTopPt"] = dqmStore_->book1D("HadronicTop_pT","Top p_{T}",30,0.,300.);
  histocontainer_[1]["HadronicTopPt"]->setAxisTitle("Hadronic Top p_{T} [GeV/c]",1);
  histocontainer_[1]["LeptonicTopPt"] = dqmStore_->book1D("LeptonicTop_pT","Top p_{T}",30,0.,300.);
  histocontainer_[1]["LeptonicTopPt"]->setAxisTitle("Leptonic Top p_{T} [GeV/c]",1);
  histocontainer_[1]["HadronicTopEta"] = dqmStore_->book1D("HadronicTop_eta","Top #eta",50,-5.,5.);
  histocontainer_[1]["HadronicTopEta"]->setAxisTitle("Hadronic Top #eta",1);
  histocontainer_[1]["LeptonicTopEta"] = dqmStore_->book1D("LeptonicTop_eta","Top #eta",50,-5.,5.);
  histocontainer_[1]["LeptonicTopEta"]->setAxisTitle("Leptonic Top #eta",1);

}

void 
SemiLeptonChecker::endJob() 
{
}
