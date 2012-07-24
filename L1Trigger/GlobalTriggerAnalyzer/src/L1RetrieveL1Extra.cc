/**
 * \class L1RetrieveL1Extra
 *
 *
 * Description: retrieve L1Extra collection, return validity flag and pointer to collection.
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
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1RetrieveL1Extra.h"

// system include files
#include <iostream>
#include <memory>
#include <string>

// user include files

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "boost/lexical_cast.hpp"


// constructor
L1RetrieveL1Extra::L1RetrieveL1Extra(const edm::ParameterSet& paramSet) :
    //
    m_tagL1ExtraMuon(paramSet.getParameter<edm::InputTag>("TagL1ExtraMuon")),
    m_tagL1ExtraIsoEG(paramSet.getParameter<edm::InputTag>("TagL1ExtraIsoEG")),
    m_tagL1ExtraNoIsoEG(paramSet.getParameter<edm::InputTag>("TagL1ExtraNoIsoEG")),
    m_tagL1ExtraCenJet(paramSet.getParameter<edm::InputTag>("TagL1ExtraCenJet")),
    m_tagL1ExtraForJet(paramSet.getParameter<edm::InputTag>("TagL1ExtraForJet")),
    m_tagL1ExtraTauJet(paramSet.getParameter<edm::InputTag>("TagL1ExtraTauJet")),
    m_tagL1ExtraEtMissMET(paramSet.getParameter<edm::InputTag>("TagL1ExtraEtMissMET")),
    m_tagL1ExtraEtMissHTM(paramSet.getParameter<edm::InputTag>("TagL1ExtraEtMissHTM")),
    m_tagL1ExtraHFRings(paramSet.getParameter<edm::InputTag>("TagL1ExtraHFRings")),
    //
    //
    m_validL1ExtraMuon(false),
    m_validL1ExtraIsoEG(false),
    m_validL1ExtraNoIsoEG(false),
    m_validL1ExtraCenJet(false),
    m_validL1ExtraForJet(false),
    m_validL1ExtraTauJet(false),
    m_validL1ExtraETT(false),
    m_validL1ExtraETM(false),
    m_validL1ExtraHTT(false),
    m_validL1ExtraHTM(false),
    m_validL1ExtraHfBitCounts(false),
    m_validL1ExtraHfRingEtSums(false),

    //
    m_l1ExtraMuon(0),
    m_l1ExtraIsoEG(0),
    m_l1ExtraNoIsoEG(0),
    m_l1ExtraCenJet(0),
    m_l1ExtraForJet(0),
    m_l1ExtraTauJet(0),
    m_l1ExtraETT(0),
    m_l1ExtraETM(0),
    m_l1ExtraHTT(0),
    m_l1ExtraHTM(0),
    m_l1ExtraHfBitCounts(0),
    m_l1ExtraHfRingEtSums(0)
    //

    {

    // empty
}

// destructor
L1RetrieveL1Extra::~L1RetrieveL1Extra() {

    // empty

}



void L1RetrieveL1Extra::retrieveL1ExtraObjects(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    //
    edm::Handle<l1extra::L1MuonParticleCollection> collL1ExtraMuon;
    iEvent.getByLabel(m_tagL1ExtraMuon, collL1ExtraMuon);

    if (collL1ExtraMuon.isValid()) {
        m_validL1ExtraMuon = true;
        m_l1ExtraMuon = collL1ExtraMuon.product();
    } else {
        LogDebug("L1RetrieveL1Extra")
                << "\n l1extra::L1MuonParticleCollection with input tag \n  "
                << m_tagL1ExtraMuon << "\n not found in the event.\n"
                << "\n Return pointer 0 and false validity tag."
                << std::endl;

        m_validL1ExtraMuon = false;
        m_l1ExtraMuon = 0;

    }

    //
    edm::Handle<l1extra::L1EmParticleCollection> collL1ExtraIsoEG;
    iEvent.getByLabel(m_tagL1ExtraIsoEG, collL1ExtraIsoEG);

    if (collL1ExtraIsoEG.isValid()) {
        m_validL1ExtraIsoEG = true;
        m_l1ExtraIsoEG = collL1ExtraIsoEG.product();
    } else {
        LogDebug("L1RetrieveL1Extra")
                << "\n l1extra::L1EmParticleCollection with input tag \n  "
                << m_tagL1ExtraIsoEG << "\n not found in the event.\n"
                << "\n Return pointer 0 and false validity tag."
                << std::endl;

        m_validL1ExtraIsoEG = false;
        m_l1ExtraIsoEG = 0;

    }

    edm::Handle<l1extra::L1EmParticleCollection> collL1ExtraNoIsoEG;
    iEvent.getByLabel(m_tagL1ExtraNoIsoEG, collL1ExtraNoIsoEG);

    if (collL1ExtraNoIsoEG.isValid()) {
        m_validL1ExtraNoIsoEG = true;
        m_l1ExtraNoIsoEG = collL1ExtraNoIsoEG.product();
    } else {
        LogDebug("L1RetrieveL1Extra")
                << "\n l1extra::L1EmParticleCollection with input tag \n  "
                << m_tagL1ExtraNoIsoEG << "\n not found in the event.\n"
                << "\n Return pointer 0 and false validity tag."
                << std::endl;

        m_validL1ExtraNoIsoEG = false;
        m_l1ExtraNoIsoEG = 0;

    }

    //
    edm::Handle<l1extra::L1JetParticleCollection> collL1ExtraCenJet;
    iEvent.getByLabel(m_tagL1ExtraCenJet, collL1ExtraCenJet);

    if (collL1ExtraCenJet.isValid()) {
        m_validL1ExtraCenJet = true;
        m_l1ExtraCenJet = collL1ExtraCenJet.product();
    } else {
        LogDebug("L1RetrieveL1Extra")
                << "\n l1extra::L1JetParticleCollection with input tag \n  "
                << m_tagL1ExtraCenJet << "\n not found in the event.\n"
                << "\n Return pointer 0 and false validity tag."
                << std::endl;

        m_validL1ExtraCenJet = false;
        m_l1ExtraCenJet = 0;

    }

    edm::Handle<l1extra::L1JetParticleCollection> collL1ExtraForJet;
    iEvent.getByLabel(m_tagL1ExtraForJet, collL1ExtraForJet);

    if (collL1ExtraForJet.isValid()) {
        m_validL1ExtraForJet = true;
        m_l1ExtraForJet = collL1ExtraForJet.product();
    } else {
        LogDebug("L1RetrieveL1Extra")
                << "\n l1extra::L1JetParticleCollection with input tag \n  "
                << m_tagL1ExtraForJet << "\n not found in the event.\n"
                << "\n Return pointer 0 and false validity tag."
                << std::endl;

        m_validL1ExtraForJet = false;
        m_l1ExtraForJet = 0;

    }

    edm::Handle<l1extra::L1JetParticleCollection> collL1ExtraTauJet;
    iEvent.getByLabel(m_tagL1ExtraTauJet, collL1ExtraTauJet);

    if (collL1ExtraTauJet.isValid()) {
        m_validL1ExtraTauJet = true;
        m_l1ExtraTauJet = collL1ExtraTauJet.product();
    } else {
        LogDebug("L1RetrieveL1Extra")
                << "\n l1extra::L1JetParticleCollection with input tag \n  "
                << m_tagL1ExtraTauJet << "\n not found in the event.\n"
                << "\n Return pointer 0 and false validity tag."
                << std::endl;

        m_validL1ExtraTauJet = false;
        m_l1ExtraTauJet = 0;

    }

    //
    edm::Handle<l1extra::L1EtMissParticleCollection> collL1ExtraEtMissMET;
    iEvent.getByLabel(m_tagL1ExtraEtMissMET, collL1ExtraEtMissMET);

    if (collL1ExtraEtMissMET.isValid()) {
        m_validL1ExtraETT = true;
        m_validL1ExtraETM = true;
        m_l1ExtraETT = collL1ExtraEtMissMET.product();
        m_l1ExtraETM = collL1ExtraEtMissMET.product();
    } else {
        LogDebug("L1RetrieveL1Extra")
                << "\n l1extra::L1EtMissParticleCollection with input tag \n  "
                << m_tagL1ExtraEtMissMET << "\n not found in the event.\n"
                << "\n Return pointer 0 and false validity tag."
                << std::endl;

        m_validL1ExtraETT = false;
        m_validL1ExtraETM = false;
        m_l1ExtraETT = 0;
        m_l1ExtraETM = 0;

    }

    edm::Handle<l1extra::L1EtMissParticleCollection> collL1ExtraEtMissHTM;
    iEvent.getByLabel(m_tagL1ExtraEtMissHTM, collL1ExtraEtMissHTM);

    if (collL1ExtraEtMissHTM.isValid()) {
        m_validL1ExtraHTT = true;
        m_validL1ExtraHTM = true;
        m_l1ExtraHTT = collL1ExtraEtMissHTM.product();
        m_l1ExtraHTM = collL1ExtraEtMissHTM.product();
    } else {
        LogDebug("L1RetrieveL1Extra")
                << "\n l1extra::L1EtMissParticleCollection with input tag \n  "
                << m_tagL1ExtraEtMissHTM << "\n not found in the event.\n"
                << "\n Return pointer 0 and false validity tag."
                << std::endl;

        m_validL1ExtraHTT = false;
        m_validL1ExtraHTM = false;
        m_l1ExtraHTT = 0;
        m_l1ExtraHTM = 0;

    }

    //
    edm::Handle<l1extra::L1HFRingsCollection> collL1ExtraHFRings;
    iEvent.getByLabel(m_tagL1ExtraHFRings, collL1ExtraHFRings);

    if (collL1ExtraHFRings.isValid()) {
        m_validL1ExtraHfBitCounts = true;
        m_validL1ExtraHfRingEtSums = true;
        m_l1ExtraHfBitCounts = collL1ExtraHFRings.product();
        m_l1ExtraHfRingEtSums = collL1ExtraHFRings.product();
    } else {
        LogDebug("L1RetrieveL1Extra")
                << "\n l1extra::L1HFRingsCollection with input tag \n  "
                << m_tagL1ExtraHFRings << "\n not found in the event.\n"
                << "\n Return pointer 0 and false validity tag."
                << std::endl;

        m_validL1ExtraHfBitCounts = false;
        m_validL1ExtraHfRingEtSums = false;
        m_l1ExtraHfBitCounts = 0;
        m_l1ExtraHfRingEtSums = 0;

    }

}
