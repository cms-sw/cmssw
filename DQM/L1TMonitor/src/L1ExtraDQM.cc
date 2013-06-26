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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "DQM/L1TMonitor/interface/L1ExtraDQM.h"

// system include files
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"


// constructor
L1ExtraDQM::L1ExtraDQM(const edm::ParameterSet& paramSet) :
    //
    m_retrieveL1Extra(paramSet.getParameter<edm::ParameterSet>("L1ExtraInputTags")),
    m_dirName(paramSet.getUntrackedParameter("DirName", std::string(
                    "L1T/L1ExtraDQM"))),
    //
    m_nrBxInEventGmt(paramSet.getParameter<int>("NrBxInEventGmt")),
    m_nrBxInEventGct(paramSet.getParameter<int>("NrBxInEventGct")),
    //
    m_dbe(0), m_resetModule(true), m_currentRun(-99),
    //
    m_nrEvJob(0),
    m_nrEvRun(0)

    {


    //
    if ((m_nrBxInEventGmt > 0) && ((m_nrBxInEventGmt % 2) == 0)) {
        m_nrBxInEventGmt = m_nrBxInEventGmt - 1;

        edm::LogInfo("L1ExtraDQM")
                << "\nWARNING: Number of bunch crossing to be monitored for GMT rounded to: "
                << m_nrBxInEventGmt
                << "\n         The number must be an odd number!\n"
                << std::endl;
    }

    if ((m_nrBxInEventGct > 0) && ((m_nrBxInEventGct % 2) == 0)) {
        m_nrBxInEventGct = m_nrBxInEventGct - 1;

        edm::LogInfo("L1ExtraDQM")
                << "\nWARNING: Number of bunch crossing to be monitored for GCT rounded to: "
                << m_nrBxInEventGct
                << "\n         The number must be an odd number!\n"
                << std::endl;
    }

    //
    m_meAnalysisL1ExtraMuon.reserve(m_nrBxInEventGmt);
    m_meAnalysisL1ExtraIsoEG.reserve(m_nrBxInEventGct);
    m_meAnalysisL1ExtraNoIsoEG.reserve(m_nrBxInEventGct);
    m_meAnalysisL1ExtraCenJet.reserve(m_nrBxInEventGct);
    m_meAnalysisL1ExtraForJet.reserve(m_nrBxInEventGct);
    m_meAnalysisL1ExtraTauJet.reserve(m_nrBxInEventGct);
    m_meAnalysisL1ExtraETT.reserve(m_nrBxInEventGct);
    m_meAnalysisL1ExtraETM.reserve(m_nrBxInEventGct);
    m_meAnalysisL1ExtraHTT.reserve(m_nrBxInEventGct);
    m_meAnalysisL1ExtraHTM.reserve(m_nrBxInEventGct);
    m_meAnalysisL1ExtraHfBitCounts.reserve(m_nrBxInEventGct);
    m_meAnalysisL1ExtraHfRingEtSums.reserve(m_nrBxInEventGct);

    m_dbe = edm::Service<DQMStore>().operator->();
    if (m_dbe == 0) {
        edm::LogInfo("L1ExtraDQM") << "\n Unable to get DQMStore service.";
    } else {

        if (paramSet.getUntrackedParameter<bool> ("DQMStore", false)) {
            m_dbe->setVerbose(0);
        }

        m_dbe->setCurrentFolder(m_dirName);

    }

}

// destructor
L1ExtraDQM::~L1ExtraDQM() {

    // empty

}

void L1ExtraDQM::analyzeL1ExtraMuon(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool bookEta = true;
    bool bookPhi = true;

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGmt; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGmt] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGmt + 1) / 2
                - m_nrBxInEventGmt;

        (m_meAnalysisL1ExtraMuon.at(iBxInEvent))->fillNrObjects(
                m_retrieveL1Extra.l1ExtraMuon(),
                m_retrieveL1Extra.validL1ExtraMuon(), isL1Coll, bxInEvent);
        (m_meAnalysisL1ExtraMuon.at(iBxInEvent))->fillPtPhiEta(
                m_retrieveL1Extra.l1ExtraMuon(),
                m_retrieveL1Extra.validL1ExtraMuon(), bookPhi, bookEta,
                isL1Coll, bxInEvent);

    }

}

void L1ExtraDQM::analyzeL1ExtraIsoEG(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool bookEta = true;
    bool bookPhi = true;

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        (m_meAnalysisL1ExtraIsoEG.at(iBxInEvent))->fillNrObjects(
                m_retrieveL1Extra.l1ExtraIsoEG(),
                m_retrieveL1Extra.validL1ExtraIsoEG(), isL1Coll, bxInEvent);
        (m_meAnalysisL1ExtraIsoEG.at(iBxInEvent))->fillPtPhiEta(
                m_retrieveL1Extra.l1ExtraIsoEG(),
                m_retrieveL1Extra.validL1ExtraIsoEG(), bookPhi, bookEta,
                isL1Coll, bxInEvent);
    }

}

void L1ExtraDQM::analyzeL1ExtraNoIsoEG(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool bookEta = true;
    bool bookPhi = true;

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        (m_meAnalysisL1ExtraNoIsoEG.at(iBxInEvent))->fillNrObjects(
                m_retrieveL1Extra.l1ExtraNoIsoEG(),
                m_retrieveL1Extra.validL1ExtraNoIsoEG(), isL1Coll, bxInEvent);
        (m_meAnalysisL1ExtraNoIsoEG.at(iBxInEvent))->fillPtPhiEta(
                m_retrieveL1Extra.l1ExtraNoIsoEG(),
                m_retrieveL1Extra.validL1ExtraNoIsoEG(), bookPhi, bookEta,
                isL1Coll, bxInEvent);
    }

}

void L1ExtraDQM::analyzeL1ExtraCenJet(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool bookEta = true;
    bool bookPhi = true;

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        (m_meAnalysisL1ExtraCenJet.at(iBxInEvent))->fillNrObjects(
                m_retrieveL1Extra.l1ExtraCenJet(),
                m_retrieveL1Extra.validL1ExtraCenJet(), isL1Coll, bxInEvent);
        (m_meAnalysisL1ExtraCenJet.at(iBxInEvent))->fillEtPhiEta(
                m_retrieveL1Extra.l1ExtraCenJet(),
                m_retrieveL1Extra.validL1ExtraCenJet(), bookPhi, bookEta,
                isL1Coll, bxInEvent);
    }

}


void L1ExtraDQM::analyzeL1ExtraForJet(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool bookPhi = true;
    bool bookEta = true;

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        (m_meAnalysisL1ExtraForJet.at(iBxInEvent))->fillNrObjects(
                m_retrieveL1Extra.l1ExtraForJet(),
                m_retrieveL1Extra.validL1ExtraForJet(), isL1Coll, bxInEvent);
        (m_meAnalysisL1ExtraForJet.at(iBxInEvent))->fillEtPhiEta(
                m_retrieveL1Extra.l1ExtraForJet(),
                m_retrieveL1Extra.validL1ExtraForJet(), bookPhi, bookEta,
                isL1Coll, bxInEvent);
    }

}

void L1ExtraDQM::analyzeL1ExtraTauJet(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool bookPhi = true;
    bool bookEta = true;

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        (m_meAnalysisL1ExtraTauJet.at(iBxInEvent))->fillNrObjects(
                m_retrieveL1Extra.l1ExtraTauJet(),
                m_retrieveL1Extra.validL1ExtraTauJet(), isL1Coll, bxInEvent);
        (m_meAnalysisL1ExtraTauJet.at(iBxInEvent))->fillEtPhiEta(
                m_retrieveL1Extra.l1ExtraTauJet(),
                m_retrieveL1Extra.validL1ExtraTauJet(), bookPhi, bookEta,
                isL1Coll, bxInEvent);
    }

}

void L1ExtraDQM::analyzeL1ExtraETT(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        (m_meAnalysisL1ExtraETT.at(iBxInEvent))->fillEtTotal(m_retrieveL1Extra.l1ExtraETT(),
                m_retrieveL1Extra.validL1ExtraETT(), isL1Coll, bxInEvent);

    }

}

void L1ExtraDQM::analyzeL1ExtraETM(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool bookPhi = true;
    bool bookEta = false;

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        (m_meAnalysisL1ExtraETM.at(iBxInEvent))->fillEtPhiEta(m_retrieveL1Extra.l1ExtraETM(),
                m_retrieveL1Extra.validL1ExtraETM(), bookPhi, bookEta,
                isL1Coll, bxInEvent);

    }

}

void L1ExtraDQM::analyzeL1ExtraHTT(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        (m_meAnalysisL1ExtraHTT.at(iBxInEvent))->fillEtTotal(m_retrieveL1Extra.l1ExtraHTT(),
                m_retrieveL1Extra.validL1ExtraHTT(), isL1Coll, bxInEvent);

    }
}

void L1ExtraDQM::analyzeL1ExtraHTM(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool bookPhi = true;
    bool bookEta = false;

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        (m_meAnalysisL1ExtraHTM.at(iBxInEvent))->fillEtPhiEta(m_retrieveL1Extra.l1ExtraHTM(),
                m_retrieveL1Extra.validL1ExtraHTM(), bookPhi, bookEta,
                isL1Coll, bxInEvent);
    }

}

void L1ExtraDQM::analyzeL1ExtraHfBitCounts(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        for (int iCount = 0; iCount < l1extra::L1HFRings::kNumRings; ++iCount) {
            (m_meAnalysisL1ExtraHfBitCounts.at(iBxInEvent))->fillHfBitCounts(
                    m_retrieveL1Extra.l1ExtraHfBitCounts(),
                    m_retrieveL1Extra.validL1ExtraHfBitCounts(), iCount,
                    isL1Coll, bxInEvent);
        }
    }

}

void L1ExtraDQM::analyzeL1ExtraHfRingEtSums(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    bool isL1Coll = true;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;

        for (int iCount = 0; iCount < l1extra::L1HFRings::kNumRings; ++iCount) {
            (m_meAnalysisL1ExtraHfRingEtSums.at(iBxInEvent))->fillHfRingEtSums(
                    m_retrieveL1Extra.l1ExtraHfRingEtSums(),
                    m_retrieveL1Extra.validL1ExtraHfRingEtSums(), iCount,
                    isL1Coll, bxInEvent);
        }
    }

}

//
void L1ExtraDQM::beginJob() {


}


void L1ExtraDQM::beginRun(const edm::Run& iRun, const edm::EventSetup& evSetup) {

    m_nrEvRun = 0;

    DQMStore* dbe = 0;
    dbe = edm::Service<DQMStore>().operator->();

    // clean up directory
    if (dbe) {
        dbe->setCurrentFolder(m_dirName);
        if (dbe->dirExists(m_dirName)) {
            dbe->rmdir(m_dirName);
        }
        dbe->setCurrentFolder(m_dirName);
    }

    std::vector<L1GtObject> l1Obj;

    // define standard sets of histograms

    //
    l1Obj.clear();
    l1Obj.push_back(Mu);
    int nrMonElements = 5;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGmt; ++iBxInEvent) {

        m_meAnalysisL1ExtraMuon.push_back(new L1ExtraDQM::L1ExtraMonElement<
                l1extra::L1MuonParticleCollection>(evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraMuon.at(iBxInEvent))->bookHistograms(evSetup, m_dbe,
                "L1_Mu", l1Obj);

    }

    //
    l1Obj.clear();
    l1Obj.push_back(IsoEG);
    nrMonElements = 4;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraIsoEG.push_back(new L1ExtraDQM::L1ExtraMonElement<
                l1extra::L1EmParticleCollection>(evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraIsoEG.at(iBxInEvent))->bookHistograms(evSetup, m_dbe,
                "L1_IsoEG", l1Obj);
    }

    //
    l1Obj.clear();
    l1Obj.push_back(NoIsoEG);
    nrMonElements = 4;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraNoIsoEG.push_back(new L1ExtraDQM::L1ExtraMonElement<
                l1extra::L1EmParticleCollection>(evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraNoIsoEG.at(iBxInEvent))->bookHistograms(evSetup, m_dbe,
                "L1_NoIsoEG", l1Obj);
    }

    //
    l1Obj.clear();
    l1Obj.push_back(CenJet);
    nrMonElements = 4;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraCenJet.push_back(new L1ExtraDQM::L1ExtraMonElement<
                l1extra::L1JetParticleCollection>(evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraCenJet.at(iBxInEvent))->bookHistograms(evSetup, m_dbe,
                "L1_CenJet", l1Obj);
    }

    //
    l1Obj.clear();
    l1Obj.push_back(ForJet);

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraForJet.push_back(new L1ExtraDQM::L1ExtraMonElement<
                l1extra::L1JetParticleCollection>(evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraForJet.at(iBxInEvent))->bookHistograms(evSetup, m_dbe,
                "L1_ForJet", l1Obj);
    }

    //
    l1Obj.clear();
    l1Obj.push_back(TauJet);

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraTauJet.push_back(new L1ExtraDQM::L1ExtraMonElement<
                l1extra::L1JetParticleCollection>(evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraTauJet.at(iBxInEvent))->bookHistograms(evSetup, m_dbe,
                "L1_TauJet", l1Obj);
    }

    //
    l1Obj.clear();
    l1Obj.push_back(ETT);
    nrMonElements = 1;

    bool bookPhi = false;
    bool bookEta = false;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraETT.push_back(new L1ExtraDQM::L1ExtraMonElement<
                l1extra::L1EtMissParticleCollection>(evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraETT.at(iBxInEvent))->bookHistograms(evSetup, m_dbe,
                "L1_ETT", l1Obj, bookPhi, bookEta);
    }

    //
    l1Obj.clear();
    l1Obj.push_back(ETM);
    nrMonElements = 2;

    bookPhi = true;
    bookEta = false;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraETM.push_back(new L1ExtraDQM::L1ExtraMonElement<
                l1extra::L1EtMissParticleCollection>(evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraETM.at(iBxInEvent))->bookHistograms(evSetup, m_dbe,
                "L1_ETM", l1Obj, bookPhi, bookEta);
    }

    //
    l1Obj.clear();
    l1Obj.push_back(HTT);
    nrMonElements = 1;

    bookPhi = false;
    bookEta = false;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraHTT.push_back(new L1ExtraDQM::L1ExtraMonElement<
                l1extra::L1EtMissParticleCollection>(evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraHTT.at(iBxInEvent))->bookHistograms(evSetup, m_dbe,
                "L1_HTT", l1Obj, bookPhi, bookEta);
    }

    //
    l1Obj.clear();
    l1Obj.push_back(HTM);
    nrMonElements = 2;

    bookPhi = true;
    bookEta = false;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraHTM.push_back(new L1ExtraDQM::L1ExtraMonElement<
                l1extra::L1EtMissParticleCollection>(evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraHTM.at(iBxInEvent))->bookHistograms(evSetup, m_dbe,
                "L1_HTM", l1Obj, bookPhi, bookEta);
    }

    //
    l1Obj.clear();
    l1Obj.push_back(HfBitCounts);
    nrMonElements = 1;

    bookPhi = false;
    bookEta = false;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraHfBitCounts.push_back(
                new L1ExtraDQM::L1ExtraMonElement<l1extra::L1HFRingsCollection>(
                        evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraHfBitCounts.at(iBxInEvent))->bookHistograms(evSetup,
                m_dbe, "L1_HfBitCounts", l1Obj, bookPhi, bookEta);
    }

    //
    l1Obj.clear();
    l1Obj.push_back(HfRingEtSums);
    nrMonElements = 1;

    bookPhi = false;
    bookEta = false;

    for (int iBxInEvent = 0; iBxInEvent < m_nrBxInEventGct; ++iBxInEvent) {

        m_meAnalysisL1ExtraHfRingEtSums.push_back(
                new L1ExtraDQM::L1ExtraMonElement<l1extra::L1HFRingsCollection>(
                        evSetup, nrMonElements));

        // convert to actual convention used in the hardware
        // (from [o, m_nrBxInEventGct] -> [-X, 0, +X]
        // write it in hex [..., E, F, 0, 1, 2, ...]
        int bxInEvent = iBxInEvent + (m_nrBxInEventGct + 1) / 2
                - m_nrBxInEventGct;
        int bxInEventHex = (bxInEvent+ 16) % 16;

        std::stringstream ss;
        std::string bxInEventHexString;
        ss << std::uppercase << std::hex << bxInEventHex;
        ss >> bxInEventHexString;

        if (m_dbe) {
            dbe->setCurrentFolder(m_dirName + "/BxInEvent_"
                    + bxInEventHexString);
        }

        (m_meAnalysisL1ExtraHfRingEtSums.at(iBxInEvent))->bookHistograms(evSetup,
                m_dbe, "L1_HfRingEtSums", l1Obj, bookPhi, bookEta);
    }

}


//
void L1ExtraDQM::analyze(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

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
}


void L1ExtraDQM::endRun(const edm::Run& run, const edm::EventSetup& evSetup) {

    // delete if event setup has changed only FIXME

    for (std::vector<L1ExtraMonElement<l1extra::L1MuonParticleCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraMuon.begin(); iterME
            != m_meAnalysisL1ExtraMuon.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraMuon.clear();


    for (std::vector<L1ExtraMonElement<l1extra::L1EmParticleCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraIsoEG.begin(); iterME
            != m_meAnalysisL1ExtraIsoEG.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraIsoEG.clear();


    for (std::vector<L1ExtraMonElement<l1extra::L1EmParticleCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraNoIsoEG.begin(); iterME
            != m_meAnalysisL1ExtraNoIsoEG.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraNoIsoEG.clear();


    for (std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraCenJet.begin(); iterME
            != m_meAnalysisL1ExtraCenJet.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraCenJet.clear();

    for (std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraForJet.begin(); iterME
            != m_meAnalysisL1ExtraForJet.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraForJet.clear();

    for (std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraTauJet.begin(); iterME
            != m_meAnalysisL1ExtraTauJet.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraTauJet.clear();


    for (std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraETT.begin(); iterME
            != m_meAnalysisL1ExtraETT.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraETT.clear();

    for (std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraETM.begin(); iterME
            != m_meAnalysisL1ExtraETM.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraETM.clear();

    for (std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraHTT.begin(); iterME
            != m_meAnalysisL1ExtraHTT.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraHTT.clear();

    for (std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraHTM.begin(); iterME
            != m_meAnalysisL1ExtraHTM.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraHTM.clear();


    for (std::vector<L1ExtraMonElement<l1extra::L1HFRingsCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraHfBitCounts.begin(); iterME
            != m_meAnalysisL1ExtraHfBitCounts.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraHfBitCounts.clear();

    for (std::vector<L1ExtraMonElement<l1extra::L1HFRingsCollection>*>::iterator
            iterME = m_meAnalysisL1ExtraHfRingEtSums.begin(); iterME
            != m_meAnalysisL1ExtraHfRingEtSums.end(); ++iterME) {

        delete *iterME;

    }
    m_meAnalysisL1ExtraHfRingEtSums.clear();

    LogDebug("L1ExtraDQM") << "\n\n endRun: " << run.id()
            << "\n  Number of events analyzed in this run:       " << m_nrEvRun
            << "\n  Total number of events analyzed in this job: " << m_nrEvJob
            << "\n" << std::endl;

}

void L1ExtraDQM::endJob() {

    edm::LogInfo("L1ExtraDQM")
            << "\n\nTotal number of events analyzed in this job: " << m_nrEvJob
            << "\n" << std::endl;

    return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1ExtraDQM);
