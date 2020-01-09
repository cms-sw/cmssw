#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_InclusiveHT.h"

SUSY_HLT_InclusiveHT::SUSY_HLT_InclusiveHT(const edm::ParameterSet &ps) {
  edm::LogInfo("SUSY_HLT_InclusiveHT") << "Constructor SUSY_HLT_InclusiveHT::SUSY_HLT_InclusiveHT " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  thePfMETCollection_ = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("pfMETCollection"));
  thePfJetCollection_ = consumes<reco::PFJetCollection>(ps.getParameter<edm::InputTag>("pfJetCollection"));
  theCaloJetCollection_ = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerPathAuxiliaryForHadronic_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForHadronic");
  triggerFilter_ = ps.getParameter<edm::InputTag>("TriggerFilter");
  ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet");
  etaThrJet_ = ps.getUntrackedParameter<double>("EtaThrJet");
}

SUSY_HLT_InclusiveHT::~SUSY_HLT_InclusiveHT() {
  edm::LogInfo("SUSY_HLT_InclusiveHT") << "Destructor SUSY_HLT_InclusiveHT::~SUSY_HLT_InclusiveHT " << std::endl;
}

void SUSY_HLT_InclusiveHT::bookHistograms(DQMStore::IBooker &ibooker_, edm::Run const &, edm::EventSetup const &) {
  edm::LogInfo("SUSY_HLT_InclusiveHT") << "SUSY_HLT_InclusiveHT::bookHistograms" << std::endl;
  // book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_InclusiveHT::analyze(edm::Event const &e, edm::EventSetup const &eSetup) {
  edm::LogInfo("SUSY_HLT_InclusiveHT") << "SUSY_HLT_InclusiveHT::analyze" << std::endl;

  //-------------------------------
  //--- MET
  //-------------------------------
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  e.getByToken(thePfMETCollection_, pfMETCollection);
  if (!pfMETCollection.isValid()) {
    edm::LogError("SUSY_HLT_InclusiveHT") << "invalid collection: PFMET"
                                          << "\n";
    return;
  }
  //-------------------------------
  //--- Jets
  //-------------------------------
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  e.getByToken(thePfJetCollection_, pfJetCollection);
  if (!pfJetCollection.isValid()) {
    edm::LogError("SUSY_HLT_InclusiveHT") << "invalid collection: PFJets"
                                          << "\n";
    return;
  }
  edm::Handle<reco::CaloJetCollection> caloJetCollection;
  e.getByToken(theCaloJetCollection_, caloJetCollection);
  if (!caloJetCollection.isValid()) {
    edm::LogError("SUSY_HLT_InclusiveHT") << "invalid collection: CaloJets"
                                          << "\n";
    return;
  }

  // check what is in the menu
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_, hltresults);
  if (!hltresults.isValid()) {
    edm::LogError("SUSY_HLT_InclusiveHT") << "invalid collection: TriggerResults"
                                          << "\n";
    return;
  }

  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if (!triggerSummary.isValid()) {
    edm::LogError("SUSY_HLT_InclusiveHT") << "invalid collection: TriggerSummary"
                                          << "\n";
    return;
  }

  // get online objects
  size_t filterIndex = triggerSummary->filterIndex(triggerFilter_);
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
  if (!(filterIndex >= triggerSummary->sizeFilters())) {
    const trigger::Keys &keys = triggerSummary->filterKeys(filterIndex);
    for (size_t j = 0; j < keys.size(); ++j) {
      trigger::TriggerObject foundObject = triggerObjects[keys[j]];
      // if(foundObject.id() == 85 && foundObject.pt() > 40.0 &&
      // fabs(foundObject.eta()) < 3.0){
      //  h_triggerJetPt->Fill(foundObject.pt());
      //  h_triggerJetEta->Fill(foundObject.eta());
      //  h_triggerJetPhi->Fill(foundObject.phi());
      //}
      if (foundObject.id() == 87) {
        h_triggerMetPt->Fill(foundObject.pt());
        h_triggerMetPhi->Fill(foundObject.phi());
      }
      if (foundObject.id() == 89) {
        h_triggerHT->Fill(foundObject.pt());
      }
    }
  }

  bool hasFired = false, hasFiredAuxiliaryForHadronicLeg = false;
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

  if (hasFiredAuxiliaryForHadronicLeg || !e.isRealData()) {
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

    if (hasFired) {
      for (reco::CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin();
           i_calojet != caloJetCollection->end();
           ++i_calojet) {
        if (i_calojet->pt() < ptThrJet_)
          continue;
        if (fabs(i_calojet->eta()) > etaThrJet_)
          continue;
        h_caloJetPt->Fill(i_calojet->pt());
        h_caloJetEta->Fill(i_calojet->eta());
        h_caloJetPhi->Fill(i_calojet->phi());
        caloHT += i_calojet->pt();
      }
      for (reco::PFJetCollection::const_iterator i_pfjet = pfJetCollection->begin(); i_pfjet != pfJetCollection->end();
           ++i_pfjet) {
        if (i_pfjet->pt() < ptThrJet_)
          continue;
        if (fabs(i_pfjet->eta()) > etaThrJet_)
          continue;
        h_pfJetPt->Fill(i_pfjet->pt());
        h_pfJetEta->Fill(i_pfjet->eta());
        h_pfJetPhi->Fill(i_pfjet->phi());
      }
      h_pfMet->Fill(pfMETCollection->begin()->et());
      h_pfMetPhi->Fill(pfMETCollection->begin()->phi());
      h_pfHT->Fill(pfHT);
      h_caloHT->Fill(caloHT);

      h_pfMetTurnOn_num->Fill(pfMETCollection->begin()->et());
      h_pfHTTurnOn_num->Fill(pfHT);
    }
    // fill denominator histograms for all events, used for turn on curves
    h_pfMetTurnOn_den->Fill(pfMETCollection->begin()->et());
    h_pfHTTurnOn_den->Fill(pfHT);
  }
}

void SUSY_HLT_InclusiveHT::bookHistos(DQMStore::IBooker &ibooker_) {
  ibooker_.cd();
  ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  // offline quantities
  h_pfMet = ibooker_.book1D("pfMet", "PF Missing E_{T}; GeV", 20, 0.0, 500.0);
  h_pfMetPhi = ibooker_.book1D("pfMetPhi", "PF MET Phi", 20, -3.5, 3.5);
  h_pfHT = ibooker_.book1D("pfHT", "PF H_{T}; GeV", 30, 0.0, 1500.0);
  h_caloHT = ibooker_.book1D("caloHT", "Calo H_{T}; GeV", 30, 0.0, 1500.0);
  h_pfJetPt = ibooker_.book1D("pfJetPt", "PFJet P_{T}; GeV", 20, 0.0, 500.0);
  h_pfJetEta = ibooker_.book1D("pfJetEta", "PFJet Eta", 20, -3.0, 3.0);
  h_pfJetPhi = ibooker_.book1D("pfJetPhi", "PFJet Phi", 20, -3.5, 3.5);
  h_caloJetPt = ibooker_.book1D("caloJetPt", "CaloJet P_{T}; GeV", 20, 0.0, 500.0);
  h_caloJetEta = ibooker_.book1D("caloJetEta", "CaloJet Eta", 20, -3.0, 3.0);
  h_caloJetPhi = ibooker_.book1D("caloJetPhi", "CaloJet Phi", 20, -3.5, 3.5);

  // online quantities
  // h_triggerJetPt = ibooker_.book1D("triggerJetPt", "Trigger Jet Pt; GeV", 20,
  // 0.0, 500.0); h_triggerJetEta = ibooker_.book1D("triggerJetEta", "Trigger
  // Jet Eta", 20, -3.0, 3.0); h_triggerJetPhi =
  // ibooker_.book1D("triggerJetPhi", "Trigger Jet Phi", 20, -3.5, 3.5);
  h_triggerMetPt = ibooker_.book1D("triggerMetPt", "Trigger Met Pt; GeV", 20, 0.0, 500.0);
  h_triggerMetPhi = ibooker_.book1D("triggerMetPhi", "Trigger Met Phi", 20, -3.5, 3.5);
  h_triggerHT = ibooker_.book1D("triggerHT", "Trigger HT; GeV", 30, 0.0, 1500.0);

  // num and den hists to be divided in harvesting step to make turn on curves
  h_pfMetTurnOn_num = ibooker_.book1D("pfMetTurnOn_num", "PF MET Turn On Numerator", 20, 0.0, 500.0);
  h_pfMetTurnOn_den = ibooker_.book1D("pfMetTurnOn_den", "PF MET Turn OnDenominator", 20, 0.0, 500.0);
  h_pfHTTurnOn_num = ibooker_.book1D("pfHTTurnOn_num", "PF HT Turn On Numerator", 30, 0.0, 1500.0);
  h_pfHTTurnOn_den = ibooker_.book1D("pfHTTurnOn_den", "PF HT Turn On Denominator", 30, 0.0, 1500.0);

  ibooker_.cd();
}

// define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_InclusiveHT);
