#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_DiJet_MET.h"

SUSY_HLT_DiJet_MET::SUSY_HLT_DiJet_MET(const edm::ParameterSet &ps) {
  edm::LogInfo("SUSY_HLT_DiJet_MET") << "Constructor SUSY_HLT_DiJet_MET::SUSY_HLT_DiJet_MET " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  thePfMETCollection_ = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("pfMETCollection"));
  theCaloMETCollection_ = consumes<reco::CaloMETCollection>(ps.getParameter<edm::InputTag>("caloMETCollection"));
  thePfJetCollection_ = consumes<reco::PFJetCollection>(ps.getParameter<edm::InputTag>("pfJetCollection"));
  theCaloJetCollection_ = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  HLTProcess_ = ps.getParameter<std::string>("HLTProcess");
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerPathAuxiliaryForHadronic_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForHadronic");
  triggerFilter_ = ps.getParameter<edm::InputTag>("TriggerFilter");
  triggerJetFilter_ = ps.getParameter<edm::InputTag>("TriggerJetFilter");
  ptThrJetTrig_ = ps.getUntrackedParameter<double>("PtThrJetTrig");
  etaThrJetTrig_ = ps.getUntrackedParameter<double>("EtaThrJetTrig");
  ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet");
  etaThrJet_ = ps.getUntrackedParameter<double>("EtaThrJet");
  metCut_ = ps.getUntrackedParameter<double>("OfflineMetCut");
}

SUSY_HLT_DiJet_MET::~SUSY_HLT_DiJet_MET() {
  edm::LogInfo("SUSY_HLT_DiJet_MET") << "Destructor SUSY_HLT_DiJet_MET::~SUSY_HLT_DiJet_MET " << std::endl;
}

void SUSY_HLT_DiJet_MET::dqmBeginRun(edm::Run const &run, edm::EventSetup const &e) {
  bool changed;

  if (!fHltConfig.init(run, e, HLTProcess_, changed)) {
    edm::LogError("SUSY_HLT_DiJet_MET") << "Initialization of HLTConfigProvider failed!!";
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
    LogDebug("SUSY_HLT_DiJet_MET") << "Path not found"
                                   << "\n";
    return;
  }

  edm::LogInfo("SUSY_HLT_DiJet_MET") << "SUSY_HLT_DiJet_MET::beginRun" << std::endl;
}

void SUSY_HLT_DiJet_MET::bookHistograms(DQMStore::IBooker &ibooker_, edm::Run const &, edm::EventSetup const &) {
  edm::LogInfo("SUSY_HLT_DiJet_MET") << "SUSY_HLT_DiJet_MET::bookHistograms" << std::endl;
  // book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_DiJet_MET::analyze(edm::Event const &e, edm::EventSetup const &eSetup) {
  edm::LogInfo("SUSY_HLT_DiJet_MET") << "SUSY_HLT_DiJet_MET::analyze" << std::endl;

  //-------------------------------
  //--- MET
  //-------------------------------
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  e.getByToken(thePfMETCollection_, pfMETCollection);
  if (!pfMETCollection.isValid()) {
    edm::LogError("SUSY_HLT_DiJet_MET") << "invalid collection: PFMET"
                                        << "\n";
    return;
  }
  edm::Handle<reco::CaloMETCollection> caloMETCollection;
  e.getByToken(theCaloMETCollection_, caloMETCollection);
  if (!caloMETCollection.isValid()) {
    edm::LogError("SUSY_HLT_DiJet_MET") << "invalid collection: CaloMET"
                                        << "\n";
    return;
  }
  //-------------------------------
  //--- Jets
  //-------------------------------
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  e.getByToken(thePfJetCollection_, pfJetCollection);
  if (!pfJetCollection.isValid()) {
    edm::LogError("SUSY_HLT_DiJet_MET") << "invalid collection: PFJets"
                                        << "\n";
    return;
  }
  edm::Handle<reco::CaloJetCollection> caloJetCollection;
  e.getByToken(theCaloJetCollection_, caloJetCollection);
  if (!caloJetCollection.isValid()) {
    edm::LogError("SUSY_HLT_DiJet_MET") << "invalid collection: CaloJets"
                                        << "\n";
    return;
  }

  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_, hltresults);
  if (!hltresults.isValid()) {
    edm::LogError("SUSY_HLT_DiJet_MET") << "invalid collection: TriggerResults"
                                        << "\n";
    return;
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if (!triggerSummary.isValid()) {
    edm::LogError("SUSY_HLT_DiJet_MET") << "invalid collection: TriggerSummary"
                                        << "\n";
    return;
  }

  // get online objects

  size_t filterIndex = triggerSummary->filterIndex(triggerFilter_);
  size_t jetFilterIndex = triggerSummary->filterIndex(triggerJetFilter_);
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();

  if (!(filterIndex >= triggerSummary->sizeFilters())) {
    const trigger::Keys &keys = triggerSummary->filterKeys(filterIndex);
    for (size_t j = 0; j < keys.size(); ++j) {
      trigger::TriggerObject foundObject = triggerObjects[keys[j]];
      h_triggerMet->Fill(foundObject.pt());
      h_triggerMetPhi->Fill(foundObject.phi());
    }
  }

  std::vector<float> ptJet, etaJet, phiJet;
  if (!(jetFilterIndex >= triggerSummary->sizeFilters())) {
    const trigger::Keys &keys_jetfilter = triggerSummary->filterKeys(jetFilterIndex);
    for (size_t j = 0; j < keys_jetfilter.size(); ++j) {
      trigger::TriggerObject foundObject = triggerObjects[keys_jetfilter[j]];
      h_triggerJetPt->Fill(foundObject.pt());
      h_triggerJetEta->Fill(foundObject.eta());
      h_triggerJetPhi->Fill(foundObject.phi());
      if (foundObject.pt() > ptThrJetTrig_ && fabs(foundObject.eta()) < etaThrJetTrig_) {
        ptJet.push_back(foundObject.pt());
        etaJet.push_back(foundObject.eta());
        phiJet.push_back(foundObject.phi());
      }
    }
  }

  bool hasFired = false;
  bool hasFiredAuxiliaryForHadronicLeg = false;
  const edm::TriggerNames &trigNames = e.triggerNames(*hltresults);
  unsigned int numTriggers = trigNames.size();
  for (unsigned int hltIndex = 0; hltIndex < numTriggers; ++hltIndex) {
    if (trigNames.triggerName(hltIndex).find(triggerPath_) != std::string::npos && hltresults->wasrun(hltIndex) &&
        hltresults->accept(hltIndex))
      hasFired = true;
    if (trigNames.triggerName(hltIndex).find(triggerPathAuxiliaryForHadronic_) != std::string::npos &&
        hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex))
      hasFiredAuxiliaryForHadronicLeg = true;
  }

  if (hasFiredAuxiliaryForHadronicLeg) {
    int offlineJetCounter = 0;
    int offlineCentralJets = 0;
    int nMatch = 0, jet1Index = -1, jet2Index = -1;

    for (reco::PFJetCollection::const_iterator i_pfjet = pfJetCollection->begin(); i_pfjet != pfJetCollection->end();
         ++i_pfjet) {
      if (i_pfjet->pt() > ptThrJet_ && fabs(i_pfjet->eta()) < etaThrJet_) {
        offlineCentralJets++;
        if (offlineCentralJets == 2 && pfMETCollection->begin()->et() > metCut_)
          h_pfJet2PtTurnOn_den->Fill(i_pfjet->pt());

        for (unsigned int itrigjet = 0; itrigjet < ptJet.size(); ++itrigjet) {
          if (itrigjet < 2 &&
              (sqrt((i_pfjet->phi() - phiJet.at(itrigjet)) * (i_pfjet->phi() - phiJet.at(itrigjet)) +
                    (i_pfjet->eta() - etaJet.at(itrigjet)) * (i_pfjet->eta() - etaJet.at(itrigjet))) < 0.4)) {
            nMatch++;
            if (nMatch == 1)
              jet1Index = offlineJetCounter;
            if (nMatch == 2)
              jet2Index = offlineJetCounter;

            if (hasFired) {
              h_pfJetPt->Fill(i_pfjet->pt());
              h_pfJetEta->Fill(i_pfjet->eta());
              h_pfJetPhi->Fill(i_pfjet->phi());
              if (offlineCentralJets == 2 && pfMETCollection->begin()->et() > metCut_)
                h_pfJet2PtTurnOn_num->Fill(i_pfjet->pt());
            }

            break;
          }
        }
      }

      offlineJetCounter++;
    }

    if (hasFired) {
      h_pfMetTurnOn_num->Fill(pfMETCollection->begin()->et());
      h_pfMetPhi->Fill(pfMETCollection->begin()->phi());
      h_caloMetvsPFMet->Fill(pfMETCollection->begin()->et(), caloMETCollection->begin()->et());

      if (jet1Index > -1 && jet2Index > -1) {
        h_pfJet1Jet2DPhi->Fill(
            fabs(reco::deltaPhi(pfJetCollection->at(jet1Index).phi(), pfJetCollection->at(jet2Index).phi())));
      }
    }

    h_pfMetTurnOn_den->Fill(pfMETCollection->begin()->et());
  }

  ptJet.clear();
  etaJet.clear();
  phiJet.clear();
}

void SUSY_HLT_DiJet_MET::bookHistos(DQMStore::IBooker &ibooker_) {
  ibooker_.cd();
  ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  // offline quantities
  h_pfMetPhi = ibooker_.book1D("pfMetPhi", "PF MET Phi", 20, -3.5, 3.5);
  h_pfJetPt = ibooker_.book1D("pfJetPt", "PF Jet p_{T} (trigger-matched jets, |#eta| < 2.4); GeV", 20, 0.0, 500.0);
  h_pfJetEta = ibooker_.book1D("pfJetEta", "PF Jet #eta (trigger-matched jets, |#eta| < 2.4)", 20, -3.0, 3.0);
  h_pfJetPhi = ibooker_.book1D("pfJetPhi", "PF Jet #phi (trigger-matched jets, |#eta| < 2.4)", 20, -3.5, 3.5);
  h_pfJet1Jet2DPhi =
      ibooker_.book1D("pfJet1Jet2DPhi", "|#Delta#phi| between two leading trigger-matched jets", 20, 0.0, 3.5);
  h_caloMetvsPFMet = ibooker_.book2D("caloMetvsPFMet", "Calo MET vs PF MET; GeV; GeV", 25, 0.0, 500.0, 25, 0.0, 500.0);

  // online quantities
  h_triggerMet = ibooker_.book1D("triggerMet", "Trigger MET; GeV", 20, 0.0, 500.0);
  h_triggerMetPhi = ibooker_.book1D("triggerMetPhi", "Trigger MET Phi", 20, -3.5, 3.5);
  h_triggerJetPt = ibooker_.book1D("triggerJetPt", "Trigger Jet p_{T}; GeV", 20, 0.0, 500.0);
  h_triggerJetEta = ibooker_.book1D("triggerJetEta", "Trigger Jet Eta", 20, -3.0, 3.0);
  h_triggerJetPhi = ibooker_.book1D("triggerJetPhi", "Trigger Jet Phi", 20, -3.5, 3.5);

  // num and den hists to be divided in harvesting step to make turn on curves
  h_pfMetTurnOn_num = ibooker_.book1D("pfMetTurnOn_num", "PF MET", 20, 0.0, 500.0);
  h_pfMetTurnOn_den = ibooker_.book1D("pfMetTurnOn_den", "PF MET Turn On Denominator", 20, 0.0, 500.0);
  h_pfJet2PtTurnOn_num =
      ibooker_.book1D("pfJet2PtTurnOn_num", "PF Jet2 Pt (NCentralPFJets >= 2, PFMET > 250)", 20, 0.0, 500.0);
  h_pfJet2PtTurnOn_den = ibooker_.book1D("pfJet2PtTurnOn_den", "PF Jet2 Pt Turn On Denominator", 20, 0.0, 500.0);

  ibooker_.cd();
}

// define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_DiJet_MET);
