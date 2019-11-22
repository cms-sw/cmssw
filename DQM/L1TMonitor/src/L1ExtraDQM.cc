/**
 * \class L1ExtraDQM
 *
 *
 * Description: online DQM module for L1Extra trigger objects.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

// this class header
#include "DQM/L1TMonitor/interface/L1ExtraDQM.h"

// system include files
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>

// constructor
L1ExtraDQM::L1ExtraDQM(const edm::ParameterSet& paramSet)
    :  //
      m_retrieveL1Extra(paramSet.getParameter<edm::ParameterSet>("L1ExtraInputTags"), consumesCollector()),
      L1ExtraIsoTauJetSource(paramSet.getParameter<edm::InputTag>("L1ExtraIsoTauJetSource_")),
      m_dirName(paramSet.getParameter<std::string>("DirName")),
      m_stage1_layer2_(paramSet.getParameter<bool>("stage1_layer2_")),
      //
      m_nrBxInEventGmt(paramSet.getParameter<int>("NrBxInEventGmt")),
      m_nrBxInEventGct(paramSet.getParameter<int>("NrBxInEventGct")),
      //
      m_resetModule(true),
      m_currentRun(-99),
      //
      m_nrEvJob(0),
      m_nrEvRun(0)

{
  //
  if ((m_nrBxInEventGmt > 0) && ((m_nrBxInEventGmt % 2) == 0)) {
    m_nrBxInEventGmt = m_nrBxInEventGmt - 1;

    edm::LogInfo("L1ExtraDQM") << "\nWARNING: Number of bunch crossing to be monitored for GMT rounded to: "
                               << m_nrBxInEventGmt << "\n         The number must be an odd number!\n"
                               << std::endl;
  }

  if ((m_nrBxInEventGct > 0) && ((m_nrBxInEventGct % 2) == 0)) {
    m_nrBxInEventGct = m_nrBxInEventGct - 1;

    edm::LogInfo("L1ExtraDQM") << "\nWARNING: Number of bunch crossing to be monitored for GCT rounded to: "
                               << m_nrBxInEventGct << "\n         The number must be an odd number!\n"
                               << std::endl;
  }

  if (m_stage1_layer2_ == true) {
    m_tagL1ExtraIsoTauJetTok =
        consumes<l1extra::L1JetParticleCollection>(paramSet.getParameter<edm::InputTag>("L1ExtraIsoTauJetSource_"));
  }
  //
  m_meAnalysisL1ExtraMuon.reserve(m_nrBxInEventGmt);
  m_meAnalysisL1ExtraIsoEG.reserve(m_nrBxInEventGct);
  m_meAnalysisL1ExtraNoIsoEG.reserve(m_nrBxInEventGct);
  m_meAnalysisL1ExtraCenJet.reserve(m_nrBxInEventGct);
  m_meAnalysisL1ExtraForJet.reserve(m_nrBxInEventGct);
  m_meAnalysisL1ExtraTauJet.reserve(m_nrBxInEventGct);
  if (m_stage1_layer2_ == true) {
    m_meAnalysisL1ExtraIsoTauJet.reserve(m_nrBxInEventGct);
  }
  m_meAnalysisL1ExtraETT.reserve(m_nrBxInEventGct);
  m_meAnalysisL1ExtraETM.reserve(m_nrBxInEventGct);
  m_meAnalysisL1ExtraHTT.reserve(m_nrBxInEventGct);
  m_meAnalysisL1ExtraHTM.reserve(m_nrBxInEventGct);
  m_meAnalysisL1ExtraHfBitCounts.reserve(m_nrBxInEventGct);
  m_meAnalysisL1ExtraHfRingEtSums.reserve(m_nrBxInEventGct);
}

// destructor
L1ExtraDQM::~L1ExtraDQM() {
  // empty
}

void L1ExtraDQM::analyzeL1ExtraMuon(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool bookEta = true;
  bool bookPhi = true;

  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGmt; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGmt + 1) / 2 - m_nrBxInEventGmt;

    (m_meAnalysisL1ExtraMuon.at(iBxInEvent))
        ->fillNrObjects(m_retrieveL1Extra.l1ExtraMuon(), m_retrieveL1Extra.validL1ExtraMuon(), isL1Coll, bxInEvent);
    (m_meAnalysisL1ExtraMuon.at(iBxInEvent))
        ->fillPtPhiEta(m_retrieveL1Extra.l1ExtraMuon(),
                       m_retrieveL1Extra.validL1ExtraMuon(),
                       bookPhi,
                       bookEta,
                       isL1Coll,
                       bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraIsoEG(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool bookEta = true;
  bool bookPhi = true;

  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    (m_meAnalysisL1ExtraIsoEG.at(iBxInEvent))
        ->fillNrObjects(m_retrieveL1Extra.l1ExtraIsoEG(), m_retrieveL1Extra.validL1ExtraIsoEG(), isL1Coll, bxInEvent);
    (m_meAnalysisL1ExtraIsoEG.at(iBxInEvent))
        ->fillPtPhiEta(m_retrieveL1Extra.l1ExtraIsoEG(),
                       m_retrieveL1Extra.validL1ExtraIsoEG(),
                       bookPhi,
                       bookEta,
                       isL1Coll,
                       bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraNoIsoEG(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool bookEta = true;
  bool bookPhi = true;

  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    (m_meAnalysisL1ExtraNoIsoEG.at(iBxInEvent))
        ->fillNrObjects(
            m_retrieveL1Extra.l1ExtraNoIsoEG(), m_retrieveL1Extra.validL1ExtraNoIsoEG(), isL1Coll, bxInEvent);
    (m_meAnalysisL1ExtraNoIsoEG.at(iBxInEvent))
        ->fillPtPhiEta(m_retrieveL1Extra.l1ExtraNoIsoEG(),
                       m_retrieveL1Extra.validL1ExtraNoIsoEG(),
                       bookPhi,
                       bookEta,
                       isL1Coll,
                       bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraCenJet(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool bookEta = true;
  bool bookPhi = true;

  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    (m_meAnalysisL1ExtraCenJet.at(iBxInEvent))
        ->fillNrObjects(m_retrieveL1Extra.l1ExtraCenJet(), m_retrieveL1Extra.validL1ExtraCenJet(), isL1Coll, bxInEvent);
    (m_meAnalysisL1ExtraCenJet.at(iBxInEvent))
        ->fillEtPhiEta(m_retrieveL1Extra.l1ExtraCenJet(),
                       m_retrieveL1Extra.validL1ExtraCenJet(),
                       bookPhi,
                       bookEta,
                       isL1Coll,
                       bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraForJet(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool bookPhi = true;
  bool bookEta = true;

  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    (m_meAnalysisL1ExtraForJet.at(iBxInEvent))
        ->fillNrObjects(m_retrieveL1Extra.l1ExtraForJet(), m_retrieveL1Extra.validL1ExtraForJet(), isL1Coll, bxInEvent);
    (m_meAnalysisL1ExtraForJet.at(iBxInEvent))
        ->fillEtPhiEta(m_retrieveL1Extra.l1ExtraForJet(),
                       m_retrieveL1Extra.validL1ExtraForJet(),
                       bookPhi,
                       bookEta,
                       isL1Coll,
                       bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraTauJet(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool bookPhi = true;
  bool bookEta = true;

  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    (m_meAnalysisL1ExtraTauJet.at(iBxInEvent))
        ->fillNrObjects(m_retrieveL1Extra.l1ExtraTauJet(), m_retrieveL1Extra.validL1ExtraTauJet(), isL1Coll, bxInEvent);
    (m_meAnalysisL1ExtraTauJet.at(iBxInEvent))
        ->fillEtPhiEta(m_retrieveL1Extra.l1ExtraTauJet(),
                       m_retrieveL1Extra.validL1ExtraTauJet(),
                       bookPhi,
                       bookEta,
                       isL1Coll,
                       bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraIsoTauJet(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool bookPhi = true;
  bool bookEta = true;

  bool isL1Coll = true;

  bool m_validL1ExtraIsoTauJet;

  edm::Handle<l1extra::L1JetParticleCollection> collL1ExtraIsoTauJet;
  iEvent.getByToken(m_tagL1ExtraIsoTauJetTok, collL1ExtraIsoTauJet);

  const l1extra::L1JetParticleCollection* m_l1ExtraIsoTauJet;

  if (collL1ExtraIsoTauJet.isValid()) {
    m_validL1ExtraIsoTauJet = true;
    m_l1ExtraIsoTauJet = collL1ExtraIsoTauJet.product();
  } else {
    LogDebug("L1RetrieveL1Extra") << "\n l1extra::L1JetParticleCollection with input tag \n  "
                                  << "m_tagL1ExtraIsoTauJet"
                                  << "\n not found in the event.\n"
                                  << "\n Return pointer 0 and false validity tag." << std::endl;

    m_validL1ExtraIsoTauJet = false;
    m_l1ExtraIsoTauJet = nullptr;
  }

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    (m_meAnalysisL1ExtraIsoTauJet.at(iBxInEvent))
        ->fillNrObjects(m_l1ExtraIsoTauJet, m_validL1ExtraIsoTauJet, isL1Coll, bxInEvent);
    (m_meAnalysisL1ExtraIsoTauJet.at(iBxInEvent))
        ->fillEtPhiEta(m_l1ExtraIsoTauJet, m_validL1ExtraIsoTauJet, bookPhi, bookEta, isL1Coll, bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraETT(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    (m_meAnalysisL1ExtraETT.at(iBxInEvent))
        ->fillEtTotal(m_retrieveL1Extra.l1ExtraETT(), m_retrieveL1Extra.validL1ExtraETT(), isL1Coll, bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraETM(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool bookPhi = true;
  bool bookEta = false;

  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    (m_meAnalysisL1ExtraETM.at(iBxInEvent))
        ->fillEtPhiEta(
            m_retrieveL1Extra.l1ExtraETM(), m_retrieveL1Extra.validL1ExtraETM(), bookPhi, bookEta, isL1Coll, bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraHTT(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    (m_meAnalysisL1ExtraHTT.at(iBxInEvent))
        ->fillEtTotal(m_retrieveL1Extra.l1ExtraHTT(), m_retrieveL1Extra.validL1ExtraHTT(), isL1Coll, bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraHTM(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool bookPhi = true;
  bool bookEta = false;

  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    (m_meAnalysisL1ExtraHTM.at(iBxInEvent))
        ->fillEtPhiEta(
            m_retrieveL1Extra.l1ExtraHTM(), m_retrieveL1Extra.validL1ExtraHTM(), bookPhi, bookEta, isL1Coll, bxInEvent);
  }
}

void L1ExtraDQM::analyzeL1ExtraHfBitCounts(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    for (int iCount = 0; iCount < l1extra::L1HFRings::kNumRings; ++iCount) {
      (m_meAnalysisL1ExtraHfBitCounts.at(iBxInEvent))
          ->fillHfBitCounts(m_retrieveL1Extra.l1ExtraHfBitCounts(),
                            m_retrieveL1Extra.validL1ExtraHfBitCounts(),
                            iCount,
                            isL1Coll,
                            bxInEvent);
    }
  }
}

void L1ExtraDQM::analyzeL1ExtraHfRingEtSums(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  bool isL1Coll = true;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;

    for (int iCount = 0; iCount < l1extra::L1HFRings::kNumRings; ++iCount) {
      (m_meAnalysisL1ExtraHfRingEtSums.at(iBxInEvent))
          ->fillHfRingEtSums(m_retrieveL1Extra.l1ExtraHfRingEtSums(),
                             m_retrieveL1Extra.validL1ExtraHfRingEtSums(),
                             iCount,
                             isL1Coll,
                             bxInEvent);
    }
  }
}

void L1ExtraDQM::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& evSetup) {}

void L1ExtraDQM::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& evSetup) {
  m_nrEvRun = 0;

  std::vector<L1GtObject> l1Obj;
  //const edm::EventSetup& evSetup;

  // define standard sets of histograms

  //
  l1Obj.clear();
  l1Obj.push_back(Mu);
  int nrMonElements = 5;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGmt; ++iBxInEvent) {
    m_meAnalysisL1ExtraMuon.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1MuonParticleCollection>(evSetup, nrMonElements));

    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    (m_meAnalysisL1ExtraMuon.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_Mu", l1Obj);
  }

  //
  l1Obj.clear();
  l1Obj.push_back(IsoEG);
  nrMonElements = 4;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraIsoEG.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1EmParticleCollection>(evSetup, nrMonElements));

    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    (m_meAnalysisL1ExtraIsoEG.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_IsoEG", l1Obj);
  }

  //
  l1Obj.clear();
  l1Obj.push_back(NoIsoEG);
  nrMonElements = 4;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraNoIsoEG.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1EmParticleCollection>(evSetup, nrMonElements));

    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    //if (m_dbe) {
    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);
    //}

    (m_meAnalysisL1ExtraNoIsoEG.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_NoIsoEG", l1Obj);
  }

  //
  l1Obj.clear();
  l1Obj.push_back(CenJet);
  nrMonElements = 4;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraCenJet.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1JetParticleCollection>(evSetup, nrMonElements));

    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    (m_meAnalysisL1ExtraCenJet.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_CenJet", l1Obj);
  }

  //
  l1Obj.clear();
  l1Obj.push_back(ForJet);

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraForJet.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1JetParticleCollection>(evSetup, nrMonElements));

    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    (m_meAnalysisL1ExtraForJet.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_ForJet", l1Obj);
  }

  //
  l1Obj.clear();
  l1Obj.push_back(TauJet);

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraTauJet.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1JetParticleCollection>(evSetup, nrMonElements));

    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    (m_meAnalysisL1ExtraTauJet.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_TauJet", l1Obj);
  }

  if (m_stage1_layer2_ == true) {
    l1Obj.clear();
    l1Obj.push_back(TauJet);
    nrMonElements = 4;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
      m_meAnalysisL1ExtraIsoTauJet.push_back(
          new L1ExtraDQM::L1ExtraMonElement<l1extra::L1JetParticleCollection>(evSetup, nrMonElements));

      int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
      int bxInEventHex = (bxInEvent + 16) % 16;

      std::stringstream ss;
      std::string bxInEventHexString;
      ss << std::uppercase << std::hex << bxInEventHex;
      ss >> bxInEventHexString;

      ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

      (m_meAnalysisL1ExtraIsoTauJet.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_IsoTauJet", l1Obj);
    }
  }

  //
  l1Obj.clear();
  l1Obj.push_back(ETT);
  nrMonElements = 1;

  bool bookPhi = false;
  bool bookEta = false;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraETT.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1EtMissParticleCollection>(evSetup, nrMonElements));

    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    (m_meAnalysisL1ExtraETT.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_ETT", l1Obj, bookPhi, bookEta);
  }

  //
  l1Obj.clear();
  l1Obj.push_back(ETM);
  nrMonElements = 2;

  bookPhi = true;
  bookEta = false;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraETM.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1EtMissParticleCollection>(evSetup, nrMonElements));
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    (m_meAnalysisL1ExtraETM.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_ETM", l1Obj, bookPhi, bookEta);
  }

  //
  l1Obj.clear();
  l1Obj.push_back(HTT);
  nrMonElements = 1;

  bookPhi = false;
  bookEta = false;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraHTT.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1EtMissParticleCollection>(evSetup, nrMonElements));
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    (m_meAnalysisL1ExtraHTT.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_HTT", l1Obj, bookPhi, bookEta);
  }

  //
  l1Obj.clear();
  l1Obj.push_back(HTM);
  nrMonElements = 2;

  bookPhi = true;
  bookEta = false;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraHTM.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1EtMissParticleCollection>(evSetup, nrMonElements));
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    if (m_stage1_layer2_ == false) {
      (m_meAnalysisL1ExtraHTM.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_HTM", l1Obj, bookPhi, bookEta);
    } else {
      (m_meAnalysisL1ExtraHTM.at(iBxInEvent))->bookhistograms(evSetup, ibooker, "L1_HTMHTT", l1Obj, bookPhi, bookEta);
    }
  }

  //
  l1Obj.clear();
  l1Obj.push_back(HfBitCounts);
  nrMonElements = 1;

  bookPhi = false;
  bookEta = false;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraHfBitCounts.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1HFRingsCollection>(evSetup, nrMonElements));
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    (m_meAnalysisL1ExtraHfBitCounts.at(iBxInEvent))
        ->bookhistograms(evSetup, ibooker, "L1_HfBitCounts", l1Obj, bookPhi, bookEta);
  }

  //
  l1Obj.clear();
  l1Obj.push_back(HfRingEtSums);
  nrMonElements = 1;

  bookPhi = false;
  bookEta = false;

  for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {
    m_meAnalysisL1ExtraHfRingEtSums.push_back(
        new L1ExtraDQM::L1ExtraMonElement<l1extra::L1HFRingsCollection>(evSetup, nrMonElements));
    int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2 - m_nrBxInEventGct;
    int bxInEventHex = (bxInEvent + 16) % 16;

    std::stringstream ss;
    std::string bxInEventHexString;
    ss << std::uppercase << std::hex << bxInEventHex;
    ss >> bxInEventHexString;

    ibooker.setCurrentFolder(m_dirName + "/BxInEvent_" + bxInEventHexString);

    if (m_stage1_layer2_ == false) {
      (m_meAnalysisL1ExtraHfRingEtSums.at(iBxInEvent))
          ->bookhistograms(evSetup, ibooker, "L1_HfRingEtSums", l1Obj, bookPhi, bookEta);
    }
    if (m_stage1_layer2_ == true) {
      (m_meAnalysisL1ExtraHfRingEtSums.at(iBxInEvent))
          ->bookhistograms(evSetup, ibooker, "L1_IsoTau_replace_Hf", l1Obj, bookPhi, bookEta);
    }
  }
}

//
void L1ExtraDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  ++m_nrEvJob;
  ++m_nrEvRun;
  //
  m_retrieveL1Extra.retrieveL1ExtraObjects(iEvent, evSetup);
  //
  analyzeL1ExtraMuon(iEvent, evSetup);
  analyzeL1ExtraIsoEG(iEvent, evSetup);
  analyzeL1ExtraNoIsoEG(iEvent, evSetup);
  analyzeL1ExtraCenJet(iEvent, evSetup);
  analyzeL1ExtraForJet(iEvent, evSetup);
  analyzeL1ExtraTauJet(iEvent, evSetup);
  analyzeL1ExtraETT(iEvent, evSetup);
  analyzeL1ExtraETM(iEvent, evSetup);
  analyzeL1ExtraHTT(iEvent, evSetup);
  analyzeL1ExtraHTM(iEvent, evSetup);
  analyzeL1ExtraHfBitCounts(iEvent, evSetup);
  analyzeL1ExtraHfRingEtSums(iEvent, evSetup);

  if (m_stage1_layer2_ == true) {
    analyzeL1ExtraIsoTauJet(iEvent, evSetup);
  }
}

void L1ExtraDQM::dqmEndRun(const edm::Run& run, const edm::EventSetup& evSetup) {
  // delete if event setup has changed only FIXME

  for (std::vector<L1ExtraMonElement<l1extra::L1MuonParticleCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraMuon.begin();
       iterME != m_meAnalysisL1ExtraMuon.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraMuon.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1EmParticleCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraIsoEG.begin();
       iterME != m_meAnalysisL1ExtraIsoEG.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraIsoEG.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1EmParticleCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraNoIsoEG.begin();
       iterME != m_meAnalysisL1ExtraNoIsoEG.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraNoIsoEG.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraCenJet.begin();
       iterME != m_meAnalysisL1ExtraCenJet.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraCenJet.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraForJet.begin();
       iterME != m_meAnalysisL1ExtraForJet.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraForJet.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraTauJet.begin();
       iterME != m_meAnalysisL1ExtraTauJet.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraTauJet.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraETT.begin();
       iterME != m_meAnalysisL1ExtraETT.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraETT.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraETM.begin();
       iterME != m_meAnalysisL1ExtraETM.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraETM.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraHTT.begin();
       iterME != m_meAnalysisL1ExtraHTT.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraHTT.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraHTM.begin();
       iterME != m_meAnalysisL1ExtraHTM.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraHTM.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1HFRingsCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraHfBitCounts.begin();
       iterME != m_meAnalysisL1ExtraHfBitCounts.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraHfBitCounts.clear();

  for (std::vector<L1ExtraMonElement<l1extra::L1HFRingsCollection>*>::iterator iterME =
           m_meAnalysisL1ExtraHfRingEtSums.begin();
       iterME != m_meAnalysisL1ExtraHfRingEtSums.end();
       ++iterME) {
    delete *iterME;
  }
  m_meAnalysisL1ExtraHfRingEtSums.clear();

  LogDebug("L1ExtraDQM") << "\n\n endRun: " << run.id()
                         << "\n  Number of events analyzed in this run:       " << m_nrEvRun
                         << "\n  Total number of events analyzed in this job: " << m_nrEvJob << "\n"
                         << std::endl;
}

// constructor L1ExtraMonElement
template <class CollectionType>
L1ExtraDQM::L1ExtraMonElement<CollectionType>::L1ExtraMonElement(const edm::EventSetup& evSetup, const int nrElements)
    : m_indexNrObjects(-1),
      m_indexPt(-1),
      m_indexEt(-1),
      m_indexPhi(-1),
      m_indexEta(-1),
      m_indexEtTotal(-1),
      m_indexCharge(-1),
      m_indexHfBitCounts(-1),
      m_indexHfRingEtSums(-1) {
  m_monElement.reserve(nrElements);
}

// destructor L1ExtraMonElement
template <class CollectionType>
L1ExtraDQM::L1ExtraMonElement<CollectionType>::~L1ExtraMonElement() {
  //empty
}

template <class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::bookhistograms(const edm::EventSetup& evSetup,
                                                                   DQMStore::IBooker& ibooker,
                                                                   const std::string& l1ExtraObject,
                                                                   const std::vector<L1GtObject>& l1GtObj,
                                                                   const bool bookPhi,
                                                                   const bool bookEta) {
  // FIXME
  L1GtObject gtObj = l1GtObj.at(0);

  //
  std::string histName;
  std::string histTitle;
  std::string xAxisTitle;
  std::string yAxisTitle;

  std::string quantity = "";

  int indexHistogram = -1;

  if (gtObj == HfBitCounts) {
    L1GetHistLimits l1GetHistLimits(evSetup);
    const L1GetHistLimits::L1HistLimits& histLimits = l1GetHistLimits.l1HistLimits(gtObj, quantity);

    const int histNrBins = histLimits.nrBins;
    const double histMinValue = histLimits.lowerBinValue;
    const double histMaxValue = histLimits.upperBinValue;

    indexHistogram++;
    m_indexHfBitCounts = indexHistogram;

    for (int iCount = 0; iCount < l1extra::L1HFRings::kNumRings; ++iCount) {
      histName = l1ExtraObject + "_Count_" + std::to_string(iCount);
      histTitle = l1ExtraObject + ": count " + std::to_string(iCount);
      xAxisTitle = l1ExtraObject;
      yAxisTitle = "Entries";

      m_monElement.push_back(ibooker.book1D(histName, histTitle, histNrBins, histMinValue, histMaxValue));
      m_monElement[m_indexHfBitCounts + iCount]->setAxisTitle(xAxisTitle, 1);
      m_monElement[m_indexHfBitCounts + iCount]->setAxisTitle(yAxisTitle, 2);
    }

    return;
  }

  // number of objects per event
  if ((gtObj == Mu) || (gtObj == IsoEG) || (gtObj == NoIsoEG) || (gtObj == CenJet) || (gtObj == ForJet) ||
      (gtObj == TauJet)) {
    quantity = "NrObjects";

    L1GetHistLimits l1GetHistLimits(evSetup);
    const L1GetHistLimits::L1HistLimits& histLimits = l1GetHistLimits.l1HistLimits(gtObj, quantity);

    const int histNrBins = histLimits.nrBins;
    const double histMinValue = histLimits.lowerBinValue;
    const double histMaxValue = histLimits.upperBinValue;

    histName = l1ExtraObject + "_NrObjectsPerEvent";
    histTitle = l1ExtraObject + ": number of objects per event";
    xAxisTitle = "Nr_" + l1ExtraObject;
    yAxisTitle = "Entries";

    m_monElement.push_back(ibooker.book1D(histName, histTitle, histNrBins, histMinValue, histMaxValue));
    indexHistogram++;

    m_monElement[indexHistogram]->setAxisTitle(xAxisTitle, 1);
    m_monElement[indexHistogram]->setAxisTitle(yAxisTitle, 2);
    m_indexNrObjects = indexHistogram;
  }

  // transverse momentum (energy)  PT (ET) [GeV]

  quantity = "ET";
  std::string quantityLongName = " transverse energy ";

  if (gtObj == Mu) {
    quantity = "PT";
    quantityLongName = " transverse momentum ";
  }

  L1GetHistLimits l1GetHistLimits(evSetup);
  const L1GetHistLimits::L1HistLimits& histLimits = l1GetHistLimits.l1HistLimits(gtObj, quantity);

  const int histNrBinsET = histLimits.nrBins;
  const double histMinValueET = histLimits.lowerBinValue;
  const double histMaxValueET = histLimits.upperBinValue;
  const std::vector<float>& binThresholdsET = histLimits.binThresholds;

  float* binThresholdsETf;
  size_t sizeBinThresholdsET = binThresholdsET.size();
  binThresholdsETf = new float[sizeBinThresholdsET];
  copy(binThresholdsET.begin(), binThresholdsET.end(), binThresholdsETf);

  LogDebug("L1ExtraDQM") << "\n PT/ET histogram for " << l1ExtraObject << "\n histNrBinsET = " << histNrBinsET
                         << "\n histMinValueET = " << histMinValueET << "\n histMaxValueET = " << histMaxValueET
                         << "\n Last bin value represents the upper limit of the histogram" << std::endl;
  for (size_t iBin = 0; iBin < sizeBinThresholdsET; ++iBin) {
    LogTrace("L1ExtraDQM") << "Bin " << iBin << ": " << quantity << " = " << binThresholdsETf[iBin] << " GeV"
                           << std::endl;
  }

  histName = l1ExtraObject + "_" + quantity;
  histTitle = l1ExtraObject + ":" + quantityLongName + quantity + " [GeV]";
  xAxisTitle = l1ExtraObject + "_" + quantity + " [GeV]";
  yAxisTitle = "Entries";

  if (gtObj == HfRingEtSums) {
    indexHistogram++;
    m_indexHfRingEtSums = indexHistogram;

    for (int iCount = 0; iCount < l1extra::L1HFRings::kNumRings; ++iCount) {
      histName = l1ExtraObject + "_Count_" + std::to_string(iCount);
      histTitle = l1ExtraObject + ": count " + std::to_string(iCount);
      xAxisTitle = l1ExtraObject;
      yAxisTitle = "Entries";

      m_monElement.push_back(ibooker.book1D(histName, histTitle, histNrBinsET, binThresholdsETf));

      m_monElement[m_indexHfRingEtSums + iCount]->setAxisTitle(xAxisTitle, 1);
      m_monElement[m_indexHfRingEtSums + iCount]->setAxisTitle(yAxisTitle, 2);
    }

  } else {
    m_monElement.push_back(ibooker.book1D(histName, histTitle, histNrBinsET, binThresholdsETf));
    indexHistogram++;

    m_monElement[indexHistogram]->setAxisTitle(xAxisTitle, 1);
    m_monElement[indexHistogram]->setAxisTitle(yAxisTitle, 2);
    m_indexPt = indexHistogram;
    m_indexEt = indexHistogram;
    m_indexEtTotal = indexHistogram;
  }

  delete[] binThresholdsETf;

  //

  if (bookPhi) {
    quantity = "phi";

    // get limits and binning from L1Extra
    L1GetHistLimits l1GetHistLimits(evSetup);
    const L1GetHistLimits::L1HistLimits& histLimits = l1GetHistLimits.l1HistLimits(gtObj, quantity);

    const int histNrBinsPhi = histLimits.nrBins;
    const double histMinValuePhi = histLimits.lowerBinValue;
    const double histMaxValuePhi = histLimits.upperBinValue;
    const std::vector<float>& binThresholdsPhi = histLimits.binThresholds;

    float* binThresholdsPhif;
    size_t sizeBinThresholdsPhi = binThresholdsPhi.size();
    binThresholdsPhif = new float[sizeBinThresholdsPhi];
    copy(binThresholdsPhi.begin(), binThresholdsPhi.end(), binThresholdsPhif);

    LogDebug("L1ExtraDQM") << "\n phi histogram for " << l1ExtraObject << "\n histNrBinsPhi = " << histNrBinsPhi
                           << "\n histMinValuePhi = " << histMinValuePhi << "\n histMaxValuePhi = " << histMaxValuePhi
                           << "\n Last bin value represents the upper limit of the histogram" << std::endl;
    for (size_t iBin = 0; iBin < sizeBinThresholdsPhi; ++iBin) {
      LogTrace("L1ExtraDQM") << "Bin " << iBin << ": phi = " << binThresholdsPhif[iBin] << " deg" << std::endl;
    }

    histName = l1ExtraObject + "_phi";
    histTitle = l1ExtraObject + ": phi distribution ";
    xAxisTitle = l1ExtraObject + "_phi [deg]";
    yAxisTitle = "Entries";

    m_monElement.push_back(ibooker.book1D(histName, histTitle, histNrBinsPhi, histMinValuePhi, histMaxValuePhi));
    indexHistogram++;

    m_monElement[indexHistogram]->setAxisTitle(xAxisTitle, 1);
    m_monElement[indexHistogram]->setAxisTitle(yAxisTitle, 2);
    m_indexPhi = indexHistogram;

    delete[] binThresholdsPhif;
  }

  //

  if (bookEta) {
    quantity = "eta";

    // get limits and binning from L1Extra
    L1GetHistLimits l1GetHistLimits(evSetup);
    const L1GetHistLimits::L1HistLimits& histLimits = l1GetHistLimits.l1HistLimits(gtObj, quantity);

    const int histNrBinsEta = histLimits.nrBins;
    const double histMinValueEta = histLimits.lowerBinValue;
    const double histMaxValueEta = histLimits.upperBinValue;
    const std::vector<float>& binThresholdsEta = histLimits.binThresholds;

    //
    float* binThresholdsEtaf;
    size_t sizeBinThresholdsEta = binThresholdsEta.size();
    binThresholdsEtaf = new float[sizeBinThresholdsEta];
    copy(binThresholdsEta.begin(), binThresholdsEta.end(), binThresholdsEtaf);

    LogDebug("L1ExtraDQM") << "\n eta histogram for " << l1ExtraObject << "\n histNrBinsEta = " << histNrBinsEta
                           << "\n histMinValueEta = " << histMinValueEta << "\n histMaxValueEta = " << histMaxValueEta
                           << "\n Last bin value represents the upper limit of the histogram" << std::endl;
    for (size_t iBin = 0; iBin < sizeBinThresholdsEta; ++iBin) {
      LogTrace("L1ExtraDQM") << "Bin " << iBin << ": eta = " << binThresholdsEtaf[iBin] << std::endl;
    }

    histName = l1ExtraObject + "_eta";
    histTitle = l1ExtraObject + ": eta distribution ";
    xAxisTitle = l1ExtraObject + "_eta";
    yAxisTitle = "Entries";

    m_monElement.push_back(ibooker.book1D(histName, histTitle, histNrBinsEta, binThresholdsEtaf));
    indexHistogram++;

    m_monElement[indexHistogram]->setAxisTitle(xAxisTitle, 1);
    m_monElement[indexHistogram]->setAxisTitle(yAxisTitle, 2);
    m_indexEta = indexHistogram;

    delete[] binThresholdsEtaf;
  }
}

template <class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillNrObjects(const CollectionType* collType,
                                                                  const bool validColl,
                                                                  const bool isL1Coll,
                                                                  const int bxInEvent) {
  if (validColl && isL1Coll) {
    size_t collSize = 0;
    for (CIterColl iterColl = collType->begin(); iterColl != collType->end(); ++iterColl) {
      if (iterColl->bx() == bxInEvent) {
        collSize++;
      }
    }
    m_monElement[m_indexNrObjects]->Fill(collSize);
  } else {
    size_t collSize = collType->size();
    m_monElement[m_indexNrObjects]->Fill(collSize);
  }
}

template <class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillPtPhiEta(const CollectionType* collType,
                                                                 const bool validColl,
                                                                 const bool bookPhi,
                                                                 const bool bookEta,
                                                                 const bool isL1Coll,
                                                                 const int bxInEvent) {
  if (validColl) {
    for (CIterColl iterColl = collType->begin(); iterColl != collType->end(); ++iterColl) {
      if (isL1Coll && (iterColl->bx() != bxInEvent)) {
        continue;
      }

      m_monElement[m_indexPt]->Fill(iterColl->pt());

      if (bookPhi) {
        // add a very small quantity to get off the bin edge
        m_monElement[m_indexPhi]->Fill(rad2deg(iterColl->phi()) + 1.e-6);
      }

      if (bookEta) {
        m_monElement[m_indexEta]->Fill(iterColl->eta());
      }
    }
  }
}

template <class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillEtPhiEta(const CollectionType* collType,
                                                                 const bool validColl,
                                                                 const bool bookPhi,
                                                                 const bool bookEta,
                                                                 const bool isL1Coll,
                                                                 const int bxInEvent) {
  if (validColl) {
    for (CIterColl iterColl = collType->begin(); iterColl != collType->end(); ++iterColl) {
      if (isL1Coll && (iterColl->bx() != bxInEvent)) {
        continue;
      }

      m_monElement[m_indexEt]->Fill(iterColl->et());

      if (bookPhi) {
        // add a very small quantity to get off the bin edge
        m_monElement[m_indexPhi]->Fill(rad2deg(iterColl->phi()) + 1.e-6);
      }

      if (bookEta) {
        m_monElement[m_indexEta]->Fill(iterColl->eta());
      }
    }
  }
}

template <class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillEtTotal(const CollectionType* collType,
                                                                const bool validColl,
                                                                const bool isL1Coll,
                                                                const int bxInEvent) {
  if (validColl) {
    for (CIterColl iterColl = collType->begin(); iterColl != collType->end(); ++iterColl) {
      if (isL1Coll && (iterColl->bx() != bxInEvent)) {
        continue;
      }

      m_monElement[m_indexEtTotal]->Fill(iterColl->etTotal());
    }
  }
}

template <class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillCharge(const CollectionType* collType,
                                                               const bool validColl,
                                                               const bool isL1Coll,
                                                               const int bxInEvent) {
  if (validColl) {
    for (CIterColl iterColl = collType->begin(); iterColl != collType->end(); ++iterColl) {
      if (isL1Coll && (iterColl->bx() != bxInEvent)) {
        continue;
      }

      m_monElement[m_indexCharge]->Fill(iterColl->charge());
    }
  }
}

template <class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillHfBitCounts(const CollectionType* collType,
                                                                    const bool validColl,
                                                                    const int countIndex,
                                                                    const bool isL1Coll,
                                                                    const int bxInEvent) {
  if (validColl) {
    for (CIterColl iterColl = collType->begin(); iterColl != collType->end(); ++iterColl) {
      if (isL1Coll && (iterColl->bx() != bxInEvent)) {
        continue;
      }

      m_monElement[m_indexHfBitCounts + countIndex]->Fill(
          iterColl->hfBitCount((l1extra::L1HFRings::HFRingLabels)countIndex));
    }
  }
}

template <class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillHfRingEtSums(const CollectionType* collType,
                                                                     const bool validColl,
                                                                     const int countIndex,
                                                                     const bool isL1Coll,
                                                                     const int bxInEvent) {
  if (validColl) {
    for (CIterColl iterColl = collType->begin(); iterColl != collType->end(); ++iterColl) {
      if (isL1Coll && (iterColl->bx() != bxInEvent)) {
        continue;
      }

      m_monElement[m_indexHfRingEtSums + countIndex]->Fill(
          iterColl->hfEtSum((l1extra::L1HFRings::HFRingLabels)countIndex));
    }
  }
}
