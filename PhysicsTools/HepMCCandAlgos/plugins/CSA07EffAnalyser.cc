
#include "PhysicsTools/HepMCCandAlgos/interface/CSA07EffAnalyser.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/TriggerResults.h"


CSA07EffAnalyser::CSA07EffAnalyser(const edm::ParameterSet & iConfig) {
  runOnChowder_  = iConfig.getParameter<bool>("runOnChowder");
  rootFileName_  = iConfig.getParameter<std::string>("rootFileName");
  rootFile_ = new TFile(rootFileName_.c_str(), "RECREATE");
  csa07T_ = new TTree("CSA07T", "CSA07 Tree");
  csa07B_ = csa07T_->Branch("CSA07B", &csa07Info_, "procId/I:ptHat/F:filterEff/F:weight/F:trigBits[90]/I");
}


CSA07EffAnalyser::~CSA07EffAnalyser() {
//   csa07T_->Write();
   rootFile_->Write();
   rootFile_->Close();
}


void CSA07EffAnalyser::analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup) {

  edm::Handle<int> procId;
  if (runOnChowder_) {
    iEvent.getByLabel("csa07EventWeightProducer", "AlpgenProcessID", procId);
  } else {
    iEvent.getByLabel("genEventProcID", procId);
  }
  csa07Info_.procId = *procId;

  edm::Handle<double> scale;
  iEvent.getByLabel("genEventScale", scale);
  csa07Info_.ptHat = *scale;

  if (runOnChowder_) {
    csa07Info_.filterEff = -1; // not available for alpgen samples
  } else {
    edm::Handle<double> filterEff;
    iEvent.getByLabel("genEventRunInfo", "FilterEfficiency", filterEff);
    csa07Info_.filterEff = *filterEff;
  }

  edm::Handle<double> weight;
  if (runOnChowder_) {
    iEvent.getByLabel("csa07EventWeightProducer", "weight", weight);
  } else {
    iEvent.getByLabel("genEventWeight", weight);
  }
  csa07Info_.weight = *weight;


  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByLabel(edm::InputTag("TriggerResults","","HLT"), triggerResults);
  if (triggerResults->size() > 90) throw cms::Exception("CSA07EffAnalyser: hardcoded trigger-bit size must be increased!");
  for (unsigned int i = 0; i < 90; i++) {
    csa07Info_.trigBits[i] = triggerResults->accept(i);
  }

  csa07T_->Fill();

}


//define this as a plug-in
DEFINE_FWK_MODULE(CSA07EffAnalyser);
