#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_MuEle_Hadronic.h"

SUSY_HLT_MuEle_Hadronic::SUSY_HLT_MuEle_Hadronic(const edm::ParameterSet &ps) {
  edm::LogInfo("SUSY_HLT_MuEle_Hadronic")
      << "Constructor SUSY_HLT_MuEle_Hadronic::SUSY_HLT_MuEle_Hadronic " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  theMuonCollection_ = consumes<reco::MuonCollection>(ps.getParameter<edm::InputTag>("MuonCollection"));
  theElectronCollection_ = consumes<reco::GsfElectronCollection>(ps.getParameter<edm::InputTag>("ElectronCollection"));
  thePfJetCollection_ = consumes<reco::PFJetCollection>(ps.getParameter<edm::InputTag>("pfJetCollection"));
  theCaloJetCollection_ = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  HLTProcess_ = ps.getParameter<std::string>("HLTProcess");
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerPathAuxiliaryForMuEle_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForMuEle");
  triggerPathAuxiliaryForHadronic_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForHadronic");
  triggerFilter_ = ps.getParameter<edm::InputTag>("TriggerFilter");
  ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet");
  etaThrJet_ = ps.getUntrackedParameter<double>("EtaThrJet");
}

SUSY_HLT_MuEle_Hadronic::~SUSY_HLT_MuEle_Hadronic() {
  edm::LogInfo("SUSY_HLT_MuEle_Hadronic")
      << "Destructor SUSY_HLT_MuEle_Hadronic::~SUSY_HLT_MuEle_Hadronic " << std::endl;
}

void SUSY_HLT_MuEle_Hadronic::dqmBeginRun(edm::Run const &run, edm::EventSetup const &e) {
  bool changed;

  if (!fHltConfig.init(run, e, HLTProcess_, changed)) {
    edm::LogError("SUSY_HLT_MuEle_Hadronic") << "Initialization of HLTConfigProvider failed!!";
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
    LogDebug("SUSY_HLT_MuEle_Hadronic") << "Path not found"
                                        << "\n";
    return;
  }
  // std::vector<std::string> filtertags = fHltConfig.moduleLabels( triggerPath_
  // ); triggerFilter_ =
  // edm::InputTag(filtertags[filtertags.size()-1],"",fHltConfig.processName());
  // triggerFilter_ = edm::InputTag("hltPFMET120Mu5L3PreFiltered", "",
  // fHltConfig.processName());

  edm::LogInfo("SUSY_HLT_MuEle_Hadronic") << "SUSY_HLT_MuEle_Hadronic::beginRun" << std::endl;
}

void SUSY_HLT_MuEle_Hadronic::bookHistograms(DQMStore::IBooker &ibooker_, edm::Run const &, edm::EventSetup const &) {
  edm::LogInfo("SUSY_HLT_MuEle_Hadronic") << "SUSY_HLT_MuEle_Hadronic::bookHistograms" << std::endl;
  // book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_MuEle_Hadronic::analyze(edm::Event const &e, edm::EventSetup const &eSetup) {
  edm::LogInfo("SUSY_HLT_MuEle_Hadronic") << "SUSY_HLT_MuEle_Hadronic::analyze" << std::endl;

  //-------------------------------
  //--- Jets
  //-------------------------------
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  e.getByToken(thePfJetCollection_, pfJetCollection);
  if (!pfJetCollection.isValid()) {
    edm::LogError("SUSY_HLT_MuEle_Hadronic") << "invalid collection: PFJets"
                                             << "\n";
    return;
  }
  edm::Handle<reco::CaloJetCollection> caloJetCollection;
  e.getByToken(theCaloJetCollection_, caloJetCollection);
  if (!caloJetCollection.isValid()) {
    edm::LogError("SUSY_HLT_MuEle_Hadronic") << "invalid collection: CaloJets"
                                             << "\n";
    return;
  }

  //-------------------------------
  //--- Muon
  //-------------------------------
  edm::Handle<reco::MuonCollection> MuonCollection;
  e.getByToken(theMuonCollection_, MuonCollection);
  if (!MuonCollection.isValid()) {
    edm::LogError("SUSY_HLT_MuEle_Hadronic") << "invalid collection: Muons "
                                             << "\n";
    return;
  }

  //-------------------------------
  //--- Electron
  //-------------------------------
  edm::Handle<reco::GsfElectronCollection> ElectronCollection;
  e.getByToken(theElectronCollection_, ElectronCollection);
  if (!ElectronCollection.isValid()) {
    edm::LogError("SUSY_HLT_MuEle_Hadronic") << "invalid collection: Electrons "
                                             << "\n";
    return;
  }

  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_, hltresults);
  if (!hltresults.isValid()) {
    edm::LogError("SUSY_HLT_MuEle_Hadronic") << "invalid collection: TriggerResults"
                                             << "\n";
    return;
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if (!triggerSummary.isValid()) {
    edm::LogError("SUSY_HLT_MuEle_Hadronic") << "invalid collection: TriggerSummary"
                                             << "\n";
    return;
  }

  // get online objects
  std::vector<float> ptMuon, etaMuon, phiMuon, ptElectron, etaElectron, phiElectron;
  size_t filterIndexEle = triggerSummary->filterIndex(triggerFilter_);
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
  if (!(filterIndexEle >= triggerSummary->sizeFilters())) {
    const trigger::Keys &keys = triggerSummary->filterKeys(filterIndexEle);
    for (size_t j = 0; j < keys.size(); ++j) {
      trigger::TriggerObject foundObject = triggerObjects[keys[j]];
      if (foundObject.id() == trigger::TriggerElectron) {  // It's an electron

        bool same = false;
        for (unsigned int x = 0; x < ptElectron.size(); x++) {
          if (fabs(ptElectron[x] - foundObject.pt()) < 0.01 || fabs(etaElectron[x] - foundObject.eta()) < 0.001 ||
              fabs(phiElectron[x] - foundObject.phi()) < 0.001)
            same = true;
        }

        if (!same) {
          h_triggerElePt->Fill(foundObject.pt());
          h_triggerEleEta->Fill(foundObject.eta());
          h_triggerElePhi->Fill(foundObject.phi());
          ptElectron.push_back(foundObject.pt());
          etaElectron.push_back(foundObject.eta());
          phiElectron.push_back(foundObject.phi());
        }
      }
      if (foundObject.id() == trigger::TriggerMuon) {  // It's a muon

        bool same = false;
        for (unsigned int x = 0; x < ptMuon.size(); x++) {
          if (fabs(ptMuon[x] - foundObject.pt()) < 0.01)
            same = true;
        }

        if (!same) {
          h_triggerMuPt->Fill(foundObject.pt());
          h_triggerMuEta->Fill(foundObject.eta());
          h_triggerMuPhi->Fill(foundObject.phi());
          ptMuon.push_back(foundObject.pt());
          etaMuon.push_back(foundObject.eta());
          phiMuon.push_back(foundObject.phi());
        }
      }
    }
  }

  if (!ptMuon.empty() && !ptElectron.empty()) {
    math::PtEtaPhiMLorentzVectorD *mu1 = new math::PtEtaPhiMLorentzVectorD(ptMuon[0], etaMuon[0], phiMuon[0], 0.106);
    math::PtEtaPhiMLorentzVectorD *ele2 =
        new math::PtEtaPhiMLorentzVectorD(ptElectron[0], etaElectron[0], phiElectron[0], 0.0005);
    (*mu1) += (*ele2);
    h_triggerMuEleMass->Fill(mu1->M());
    delete mu1;
    delete ele2;
  } else {
    h_triggerMuEleMass->Fill(-1.);
  }

  bool hasFired = false;
  bool hasFiredAuxiliaryForMuEleLeg = false;
  bool hasFiredAuxiliaryForHadronicLeg = false;
  const edm::TriggerNames &trigNames = e.triggerNames(*hltresults);
  unsigned int numTriggers = trigNames.size();
  for (unsigned int hltIndex = 0; hltIndex < numTriggers; ++hltIndex) {
    if (trigNames.triggerName(hltIndex).find(triggerPath_) != std::string::npos && hltresults->wasrun(hltIndex) &&
        hltresults->accept(hltIndex))
      hasFired = true;
    if (trigNames.triggerName(hltIndex).find(triggerPathAuxiliaryForMuEle_) != std::string::npos &&
        hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex))
      hasFiredAuxiliaryForMuEleLeg = true;
    if (trigNames.triggerName(hltIndex).find(triggerPathAuxiliaryForHadronic_) != std::string::npos &&
        hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex))
      hasFiredAuxiliaryForHadronicLeg = true;
  }

  if (hasFiredAuxiliaryForMuEleLeg || hasFiredAuxiliaryForHadronicLeg) {
    // Matching the muon
    int indexOfMatchedMuEle[2] = {-1};
    int matchedCounter = 0;
    int offlineCounter = 0;
    for (reco::MuonCollection::const_iterator muon = MuonCollection->begin();
         (muon != MuonCollection->end() && matchedCounter < 1);
         ++muon) {
      for (size_t off_i = 0; off_i < ptMuon.size(); ++off_i) {
        if (sqrt((muon->phi() - phiMuon[off_i]) * (muon->phi() - phiMuon[off_i]) +
                 (muon->eta() - etaMuon[off_i]) * (muon->eta() - etaMuon[off_i])) < 0.5) {
          indexOfMatchedMuEle[matchedCounter] = offlineCounter;
          matchedCounter++;
          break;
        }
      }
      offlineCounter++;
    }

    for (reco::GsfElectronCollection::const_iterator electron = ElectronCollection->begin();
         (electron != ElectronCollection->end() && matchedCounter < 2);
         ++electron) {
      for (size_t off_i = 0; off_i < ptElectron.size(); ++off_i) {
        if (sqrt((electron->phi() - phiElectron[off_i]) * (electron->phi() - phiElectron[off_i]) +
                 (electron->eta() - etaElectron[off_i]) * (electron->eta() - etaElectron[off_i])) < 0.5) {
          indexOfMatchedMuEle[matchedCounter] = offlineCounter;
          matchedCounter++;
          break;
        }
      }
      offlineCounter++;
    }

    float caloHT = 0.0;
    float pfHT = 0.0;
    for (reco::PFJetCollection::const_iterator i_pfjet = pfJetCollection->begin(); i_pfjet != pfJetCollection->end();
         ++i_pfjet) {
      if (i_pfjet->pt() < ptThrJet_)
        continue;
      if (fabs(i_pfjet->eta()) > etaThrJet_)
        continue;
      pfHT += i_pfjet->pt();
    }
    for (reco::CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin();
         i_calojet != caloJetCollection->end();
         ++i_calojet) {
      if (i_calojet->pt() < ptThrJet_)
        continue;
      if (fabs(i_calojet->eta()) > etaThrJet_)
        continue;
      caloHT += i_calojet->pt();
    }

    if (hasFiredAuxiliaryForMuEleLeg && !MuonCollection->empty() && !ElectronCollection->empty()) {
      if (hasFired && indexOfMatchedMuEle[1] >= 0 && indexOfMatchedMuEle[0] >= 0) {  // check that ele is there too
        h_EleTurnOn_num->Fill(ElectronCollection->at(indexOfMatchedMuEle[1]).pt());
        h_MuTurnOn_num->Fill(MuonCollection->at(indexOfMatchedMuEle[0]).pt());
      }
      h_MuTurnOn_den->Fill(MuonCollection->at(0).pt());
      h_EleTurnOn_den->Fill(ElectronCollection->at(0).pt());
    }

    if (hasFiredAuxiliaryForHadronicLeg) {
      if (hasFired) {
        h_pfHTTurnOn_num->Fill(pfHT);
      }
      h_pfHTTurnOn_den->Fill(pfHT);
    }
  }
}

void SUSY_HLT_MuEle_Hadronic::bookHistos(DQMStore::IBooker &ibooker_) {
  ibooker_.cd();
  ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  // offline quantities

  // online quantities
  h_triggerMuPt = ibooker_.book1D("triggerMuPt", "Trigger Muon Pt; GeV", 50, 0.0, 500.0);
  h_triggerMuEta = ibooker_.book1D("triggerMuEta", "Trigger Muon Eta", 20, -3.0, 3.0);
  h_triggerMuPhi = ibooker_.book1D("triggerMuPhi", "Trigger Muon Phi", 20, -3.5, 3.5);

  h_triggerElePt = ibooker_.book1D("triggerElePt", "Trigger Electron Pt; GeV", 50, 0.0, 500.0);
  h_triggerEleEta = ibooker_.book1D("triggerEleEta", "Trigger Electron Eta", 20, -3.0, 3.0);
  h_triggerElePhi = ibooker_.book1D("triggerElePhi", "Trigger Electron Phi", 20, -3.5, 3.5);

  h_triggerMuEleMass = ibooker_.book1D("triggerMuEleMass", "Trigger MuEle Mass", 202, -2, 200);

  // num and den hists to be divided in harvesting step to make turn on curves
  h_pfHTTurnOn_num = ibooker_.book1D("pfHTTurnOn_num", "PF HT Turn On Numerator", 30, 0.0, 1500.0);
  h_pfHTTurnOn_den = ibooker_.book1D("pfHTTurnOn_den", "PF HT Turn On Denominator", 30, 0.0, 1500.0);
  h_MuTurnOn_num = ibooker_.book1D("MuTurnOn_num", "Muon Turn On Numerator", 30, 0.0, 150);
  h_MuTurnOn_den = ibooker_.book1D("MuTurnOn_den", "Muon Turn On Denominator", 30, 0.0, 150.0);
  h_EleTurnOn_num = ibooker_.book1D("EleTurnOn_num", "Electron Turn On Numerator", 30, 0.0, 150);
  h_EleTurnOn_den = ibooker_.book1D("EleTurnOn_den", "Electron Turn On Denominator", 30, 0.0, 150.0);

  ibooker_.cd();
}

// define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_MuEle_Hadronic);
