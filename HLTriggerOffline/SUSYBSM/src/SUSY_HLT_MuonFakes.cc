#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_MuonFakes.h"

SUSY_HLT_MuonFakes::SUSY_HLT_MuonFakes(const edm::ParameterSet &ps) {
  edm::LogInfo("SUSY_HLT_MuonFakes") << "Constructor SUSY_HLT_MuonFakes::SUSY_HLT_MuonFakes " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  HLTProcess_ = ps.getParameter<std::string>("HLTProcess");
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerFilter_ = ps.getParameter<edm::InputTag>("TriggerFilter");
}

SUSY_HLT_MuonFakes::~SUSY_HLT_MuonFakes() {
  edm::LogInfo("SUSY_HLT_MuonFakes") << "Destructor SUSY_HLT_MuonFakes::~SUSY_HLT_MuonFakes " << std::endl;
}

void SUSY_HLT_MuonFakes::dqmBeginRun(edm::Run const &run, edm::EventSetup const &e) {
  bool changed;

  if (!fHltConfig.init(run, e, HLTProcess_, changed)) {
    edm::LogError("SUSY_HLT_MuonFakes") << "Initialization of HLTConfigProvider failed!!";
    return;
  }

  bool pathFound = false;
  const std::vector<std::string> allTrigNames = fHltConfig.triggerNames();
  for (size_t j = 0; j < allTrigNames.size(); ++j) {
    if (allTrigNames[j].find(triggerPath_) != std::string::npos) {
      pathFound = true;
    }
  }

  if (!pathFound) {
    edm::LogInfo("SUSY_HLT_MuonFakes") << "Path not found"
                                       << "\n";
    return;
  }
  edm::LogInfo("SUSY_HLT_MuonFakes") << "SUSY_HLT_MuonFakes::beginRun" << std::endl;
}

void SUSY_HLT_MuonFakes::bookHistograms(DQMStore::IBooker &ibooker_, edm::Run const &, edm::EventSetup const &) {
  edm::LogInfo("SUSY_HLT_MuonFakes") << "SUSY_HLT_MuonFakes::bookHistograms" << std::endl;
  // book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_MuonFakes::analyze(edm::Event const &e, edm::EventSetup const &eSetup) {
  edm::LogInfo("SUSY_HLT_MuonFakes") << "SUSY_HLT_MuonFakes::analyze" << std::endl;

  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_, hltresults);
  if (!hltresults.isValid()) {
    edm::LogError("SUSY_HLT_MuonFakes") << "invalid collection: TriggerResults"
                                        << "\n";
    return;
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if (!triggerSummary.isValid()) {
    edm::LogError("SUSY_HLT_MuonFakes") << "invalid collection: TriggerSummary"
                                        << "\n";
    return;
  }

  // get online objects
  std::vector<float> ptMuon, etaMuon, phiMuon;
  size_t filterIndex = triggerSummary->filterIndex(triggerFilter_);
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
  if (!(filterIndex >= triggerSummary->sizeFilters())) {
    const trigger::Keys &keys = triggerSummary->filterKeys(filterIndex);
    for (size_t j = 0; j < keys.size(); ++j) {
      trigger::TriggerObject foundObject = triggerObjects[keys[j]];
      if (foundObject.id() == 13) {  // Muons check number
        h_triggerMuPt->Fill(foundObject.pt());
        h_triggerMuEta->Fill(foundObject.eta());
        h_triggerMuPhi->Fill(foundObject.phi());
        ptMuon.push_back(foundObject.pt());
        etaMuon.push_back(foundObject.eta());
        phiMuon.push_back(foundObject.phi());
      }
    }
  }

  //  bool hasFired = false;
  //  const edm::TriggerNames& trigNames = e.triggerNames(*hltresults);
  //  unsigned int numTriggers = trigNames.size();
  //  for( unsigned int hltIndex=0; hltIndex<numTriggers; ++hltIndex ){
  //    if (trigNames.triggerName(hltIndex)==triggerPath_ &&
  //    hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFired =
  //    true;
  //  }
}

void SUSY_HLT_MuonFakes::bookHistos(DQMStore::IBooker &ibooker_) {
  ibooker_.cd();
  ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  // online quantities
  h_triggerMuPt = ibooker_.book1D("triggerMuPt", "Trigger Mu Pt; GeV", 40, 0.0, 80.0);
  h_triggerMuEta = ibooker_.book1D("triggerMuEta", "Trigger Mu Eta", 20, -2.5, 2.5);
  h_triggerMuPhi = ibooker_.book1D("triggerMuPhi", "Trigger Mu Phi", 20, -3.5, 3.5);

  // num and den hists to be divided in harvesting step to make turn on curves
  ibooker_.cd();
}

// define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_MuonFakes);
