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

/// input tag for a given collection
const edm::InputTag L1RetrieveL1Extra::inputTagL1ExtraColl(const L1GtObject& gtObject) const {

    edm::InputTag emptyInputTag;

    switch (gtObject) {

        case Mu: {
            return m_tagL1ExtraMuon;
        }
            break;

        case NoIsoEG: {
            return m_tagL1ExtraNoIsoEG;
        }
            break;

        case IsoEG: {
            return m_tagL1ExtraIsoEG;
        }
            break;

        case CenJet: {
            return m_tagL1ExtraCenJet;
        }
            break;

        case ForJet: {
            return m_tagL1ExtraForJet;
        }
            break;

        case TauJet: {
            return m_tagL1ExtraTauJet;
        }
            break;

        case ETM:
        case ETT: {
            return m_tagL1ExtraEtMissMET;
        }
            break;

        case HTT:
        case HTM: {
            return m_tagL1ExtraEtMissHTM;
        }
            break;

        case JetCounts: {
            // TODO update when JetCounts will be available
            return emptyInputTag;
        }
            break;

        case HfBitCounts:
        case HfRingEtSums: {
            return m_tagL1ExtraHFRings;
        }
            break;

        case TechTrig: {
            return emptyInputTag;
        }
            break;

        case Castor: {
            return emptyInputTag;
        }
            break;

        case BPTX: {
            return emptyInputTag;
        }
            break;

        case GtExternal: {
            return emptyInputTag;
        }
            break;

        case ObjNull: {
            return emptyInputTag;
        }
            break;

        default: {
            edm::LogInfo("L1GtObject") << "\n  '" << gtObject
                    << "' is not a recognized L1GtObject. ";

            return emptyInputTag;

        }
            break;
    }

    return emptyInputTag;
}

const bool L1RetrieveL1Extra::validL1ExtraColl(const L1GtObject& gtObject) const {

    switch (gtObject) {

        case Mu: {
            return m_validL1ExtraMuon;
        }
            break;

        case NoIsoEG: {
            return m_validL1ExtraNoIsoEG;
        }
            break;

        case IsoEG: {
            return m_validL1ExtraIsoEG;
        }
            break;

        case CenJet: {
            return m_validL1ExtraCenJet;
        }
            break;

        case ForJet: {
            return m_validL1ExtraForJet;
        }
            break;

        case TauJet: {
            return m_validL1ExtraTauJet;
        }
            break;

        case ETM: {
            return m_validL1ExtraETM;
        }
            break;

        case ETT: {
            return m_validL1ExtraETT;
        }
            break;

        case HTT: {
            return m_validL1ExtraHTT;
        }
            break;

        case HTM: {
            return m_validL1ExtraHTM;
        }
            break;

        case JetCounts: {
            // TODO update when JetCounts will be available
            return false;
        }
            break;

        case HfBitCounts: {
            return m_validL1ExtraHfBitCounts;
        }
            break;

        case HfRingEtSums: {
            return m_validL1ExtraHfRingEtSums;
        }
            break;

        case TechTrig: {
            return false;
        }
            break;

        case Castor: {
            return false;
        }
            break;

        case BPTX: {
            return false;
        }
            break;

        case GtExternal: {
            return false;
        }
            break;

        case ObjNull: {
            return false;
        }
            break;

        default: {
            edm::LogInfo("L1GtObject") << "\n  '" << gtObject
                    << "' is not a recognized L1GtObject. ";

            return false;

        }
            break;
    }

    return false;
}

void L1RetrieveL1Extra::printL1Extra(std::ostream& oStr,
        const L1GtObject& gtObject, const bool checkBxInEvent,
        const int bxInEvent, const bool checkObjIndexInColl,
        const int objIndexInColl) const {

    if (!validL1ExtraColl(gtObject)) {
        oStr << "\n L1Extra collection for L1 GT object "
                << l1GtObjectEnumToString(gtObject)
                << " with collection input tag " << inputTagL1ExtraColl(
                gtObject) << " not valid." << std::endl;
    }

    switch (gtObject) {

        case Mu: {
            oStr << "\n Mu collection\n" << std::endl;

            int indexInColl = -1;

            for (l1extra::L1MuonParticleCollection::const_iterator iterColl =
                    m_l1ExtraMuon->begin(); iterColl
                    != m_l1ExtraMuon->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        indexInColl++;

                        if (!checkObjIndexInColl) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                    << " indexInColl = " << indexInColl
                                    << " PT = " << std::right << std::setw(6) << (iterColl->pt()) << " GeV"
                                    << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                    << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                        } else {
                            if (objIndexInColl == indexInColl) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " indexInColl = " << indexInColl
                                        << " PT = " << std::right << std::setw(6) << (iterColl->pt()) << " GeV"
                                        << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                        << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                            }
                        }
                    }
                } else {
                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx()) << " PT = "
                            << std::right << std::setw(6) << (iterColl->pt()) << " GeV" << " eta = "
                            << std::right << std::setw(8) << (iterColl->eta()) << " phi = "
                            << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;

                }

            }

        }
            break;

        case NoIsoEG: {
            oStr << "\n NoIsoEG collection\n" << std::endl;

            int indexInColl = -1;

            for (l1extra::L1EmParticleCollection::const_iterator iterColl =
                    m_l1ExtraNoIsoEG->begin(); iterColl
                    != m_l1ExtraNoIsoEG->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        indexInColl++;

                        if (!checkObjIndexInColl) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                    << " indexInColl = " << indexInColl
                                    << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                    << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                    << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                        } else {
                            if (objIndexInColl == indexInColl) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " indexInColl = " << indexInColl
                                        << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                        << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                        << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                            }
                        }
                    }
                } else {
                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx()) << " ET = "
                            << std::right << std::setw(6) << (iterColl->et()) << " GeV" << " eta = "
                            << std::right << std::setw(8) << (iterColl->eta()) << " phi = "
                            << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;

                }

            }
        }
            break;

        case IsoEG: {
            oStr << "\n IsoEG collection\n" << std::endl;

            int indexInColl = -1;

            for (l1extra::L1EmParticleCollection::const_iterator iterColl =
                    m_l1ExtraIsoEG->begin(); iterColl != m_l1ExtraIsoEG->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        indexInColl++;

                        if (!checkObjIndexInColl) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                    << " indexInColl = " << indexInColl
                                    << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                    << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                    << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                        } else {
                            if (objIndexInColl == indexInColl) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " indexInColl = " << indexInColl
                                        << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                        << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                        << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                            }
                        }
                    }
                } else {
                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx()) << " ET = "
                            << std::right << std::setw(6) << (iterColl->et()) << " GeV" << " eta = "
                            << std::right << std::setw(8) << (iterColl->eta()) << " phi = "
                            << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;

                }

            }
        }
            break;

        case CenJet: {
            oStr << "\n CenJet collection\n" << std::endl;

            int indexInColl = -1;

            for (l1extra::L1JetParticleCollection::const_iterator iterColl =
                    m_l1ExtraCenJet->begin(); iterColl
                    != m_l1ExtraCenJet->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        indexInColl++;

                        if (!checkObjIndexInColl) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                    << " indexInColl = " << indexInColl
                                    << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                    << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                    << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                        } else {
                            if (objIndexInColl == indexInColl) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " indexInColl = " << indexInColl
                                        << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                        << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                        << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                            }
                        }
                    }
                } else {
                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx()) << " ET = "
                            << std::right << std::setw(6) << (iterColl->et()) << " GeV" << " eta = "
                            << std::right << std::setw(8) << (iterColl->eta()) << " phi = "
                            << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;

                }

            }
        }
            break;

        case ForJet: {
            oStr << "\n ForJet collection\n" << std::endl;

            int indexInColl = -1;

            for (l1extra::L1JetParticleCollection::const_iterator iterColl =
                    m_l1ExtraForJet->begin(); iterColl
                    != m_l1ExtraForJet->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        indexInColl++;

                        if (!checkObjIndexInColl) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                    << " indexInColl = " << indexInColl
                                    << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                    << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                    << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                        } else {
                            if (objIndexInColl == indexInColl) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " indexInColl = " << indexInColl
                                        << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                        << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                        << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                            }
                        }
                    }
                } else {
                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx()) << " ET = "
                            << std::right << std::setw(6) << (iterColl->et()) << " GeV" << " eta = "
                            << std::right << std::setw(8) << (iterColl->eta()) << " phi = "
                            << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;

                }

            }
        }
            break;

        case TauJet: {
            oStr << "\n TauJet collection\n" << std::endl;

            int indexInColl = -1;

            for (l1extra::L1JetParticleCollection::const_iterator iterColl =
                    m_l1ExtraTauJet->begin(); iterColl
                    != m_l1ExtraTauJet->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        indexInColl++;

                        if (!checkObjIndexInColl) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                    << " indexInColl = " << indexInColl
                                    << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                    << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                    << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                        } else {
                            if (objIndexInColl == indexInColl) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " indexInColl = " << indexInColl
                                        << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                        << " eta = " << std::right << std::setw(8) << (iterColl->eta())
                                        << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                            }
                        }
                    }
                } else {
                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx()) << " ET = "
                            << std::right << std::setw(6) << (iterColl->et()) << " GeV" << " eta = "
                            << std::right << std::setw(8) << (iterColl->eta()) << " phi = "
                            << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;

                }

            }
        }
            break;

        case ETM: {
            oStr << "\n ETM collection\n" << std::endl;

            int indexInColl = -1;

            for (l1extra::L1EtMissParticleCollection::const_iterator iterColl =
                    m_l1ExtraETM->begin(); iterColl != m_l1ExtraETM->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        indexInColl++;

                        if (!checkObjIndexInColl) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                    << " indexInColl = " << indexInColl
                                    << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                    << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                        } else {
                            if (objIndexInColl == indexInColl) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " indexInColl = " << indexInColl
                                        << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                        << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                            }
                        }
                    }
                } else {
                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx()) << " ET = "
                            << std::right << std::setw(6) << (iterColl->et()) << " GeV" << " phi = "
                            << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;

                }

            }
        }
            break;

        case ETT: {
            oStr << "\n ETT collection\n" << std::endl;

            int indexInColl = -1;

            for (l1extra::L1EtMissParticleCollection::const_iterator iterColl =
                    m_l1ExtraETT->begin(); iterColl != m_l1ExtraETT->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        indexInColl++;

                        if (!checkObjIndexInColl) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                    << " indexInColl = " << indexInColl
                                    << " ET = " << std::right << std::setw(6) <<(iterColl->etTotal()) << " GeV" << std::endl;
                        } else {
                            if (objIndexInColl == indexInColl) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " indexInColl = " << indexInColl
                                        << " ET = " << std::right << std::setw(6) <<(iterColl->etTotal()) << " GeV" << std::endl;
                            }
                        }
                    }
                } else {
                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx()) << " ET = "
                            << std::right << std::setw(6) <<(iterColl->etTotal()) << " GeV" << std::endl;

                }

            }
        }
            break;

        case HTT: {
            oStr << "\n HTT collection\n" << std::endl;

            int indexInColl = -1;

            for (l1extra::L1EtMissParticleCollection::const_iterator iterColl =
                    m_l1ExtraHTT->begin(); iterColl != m_l1ExtraHTT->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        indexInColl++;

                        if (!checkObjIndexInColl) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                    << " indexInColl = " << indexInColl
                                    << " ET = " << std::right << std::setw(6) <<(iterColl->etTotal()) << " GeV" << std::endl;
                        } else {
                            if (objIndexInColl == indexInColl) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " indexInColl = " << indexInColl
                                        << " ET = " << std::right << std::setw(6) <<(iterColl->etTotal()) << " GeV" << std::endl;
                            }
                        }
                    }
                } else {
                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx()) << " ET = "
                            << std::right << std::setw(6) <<(iterColl->etTotal()) << " GeV" << std::endl;

                }

            }
        }
            break;

        case HTM: {
            oStr << "\n HTM collection\n" << std::endl;

            int indexInColl = -1;

            for (l1extra::L1EtMissParticleCollection::const_iterator iterColl =
                    m_l1ExtraHTM->begin(); iterColl != m_l1ExtraHTM->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        indexInColl++;

                        if (!checkObjIndexInColl) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                    << " indexInColl = " << indexInColl
                                    << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                    << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                        } else {
                            if (objIndexInColl == indexInColl) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " indexInColl = " << indexInColl
                                        << " ET = " << std::right << std::setw(6) << (iterColl->et()) << " GeV"
                                        << " phi = " << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;
                            }
                        }
                    }
                } else {
                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx()) << " ET = "
                            << std::right << std::setw(6) << (iterColl->et()) << " GeV" << " phi = "
                            << std::right << std::setw(8) << (iterColl->phi()) << " rad" << std::endl;

                }

            }
        }
            break;

        case JetCounts: {
            // TODO print if and when JetCounts will be available
        }
            break;

        case HfBitCounts: {
            oStr << "\n HfBitCounts collection\n" << std::endl;

            for (l1extra::L1HFRingsCollection::const_iterator iterColl =
                    m_l1ExtraHfBitCounts->begin(); iterColl
                    != m_l1ExtraHfBitCounts->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        if (!checkObjIndexInColl) {

                            for (int iCount = 0; iCount
                                    < l1extra::L1HFRings::kNumRings; ++iCount) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " count = " << iCount << " HF counts = "
                                        << (iterColl->hfBitCount(
                                                (l1extra::L1HFRings::HFRingLabels) iCount)) << std::endl;
                            }

                        } else {
                            for (int iCount = 0; iCount
                                    < l1extra::L1HFRings::kNumRings; ++iCount) {
                                if (objIndexInColl == iCount) {
                                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                            << " count = " << iCount
                                            << " HF counts = "
                                            << (iterColl->hfBitCount(
                                                    (l1extra::L1HFRings::HFRingLabels) iCount)) << std::endl;
                                }
                            }
                        }
                    }
                } else {
                    for (int iCount = 0; iCount < l1extra::L1HFRings::kNumRings; ++iCount) {
                        if (objIndexInColl == iCount) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx())
                                    << " count = " << iCount << " HF counts = "
                                    << (iterColl->hfBitCount(
                                            (l1extra::L1HFRings::HFRingLabels) iCount)) << std::endl;
                        }
                    }

                }

            }
        }
            break;

        case HfRingEtSums: {
            oStr << "\n HfRingEtSums collection\n" << std::endl;

            for (l1extra::L1HFRingsCollection::const_iterator iterColl =
                    m_l1ExtraHfRingEtSums->begin(); iterColl
                    != m_l1ExtraHfRingEtSums->end(); ++iterColl) {

                if (checkBxInEvent) {
                    if (iterColl->bx() != bxInEvent) {
                        continue;
                        oStr << "\n   BxInEvent " << bxInEvent
                                << ": collection not in the event" << std::endl;
                    } else {

                        if (!checkObjIndexInColl) {

                            for (int iCount = 0; iCount
                                    < l1extra::L1HFRings::kNumRings; ++iCount) {
                                oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                        << " count = " << iCount
                                        << " HF ET sum = "
                                        << (iterColl->hfEtSum(
                                                (l1extra::L1HFRings::HFRingLabels) iCount)) << " GeV" << std::endl;
                            }

                        } else {
                            for (int iCount = 0; iCount
                                    < l1extra::L1HFRings::kNumRings; ++iCount) {
                                if (objIndexInColl == iCount) {
                                    oStr << "     bxInEvent = "  << std::right << std::setw(2) << bxInEvent
                                            << " count = " << iCount
                                            << " HF ET sum = "
                                            << (iterColl->hfEtSum(
                                                    (l1extra::L1HFRings::HFRingLabels) iCount)) << " GeV" << std::endl;
                                }
                            }
                        }
                    }
                } else {
                    for (int iCount = 0; iCount < l1extra::L1HFRings::kNumRings; ++iCount) {
                        if (objIndexInColl == iCount) {
                            oStr << "     bxInEvent = "  << std::right << std::setw(2) << (iterColl->bx())
                                    << " count = " << iCount << " HF ET sum = "
                                    << (iterColl->hfEtSum(
                                            (l1extra::L1HFRings::HFRingLabels) iCount)) << " GeV" << std::endl;
                        }
                    }

                }

            }
        }
            break;

        case TechTrig: {
            // do nothing, not in L1Extra
        }
            break;

        case Castor: {
            // do nothing, not in L1Extra
        }
            break;

        case BPTX: {
            // do nothing, not in L1Extra
        }
            break;

        case GtExternal: {
            // do nothing, not in L1Extra
        }
            break;

        case ObjNull: {
            // do nothing, not in L1Extra
        }
            break;

        default: {
            edm::LogInfo("L1GtObject") << "\n  '" << gtObject
                    << "' is not a recognized L1GtObject. ";

            // do nothing

        }
            break;
    }

}

void L1RetrieveL1Extra::printL1Extra(std::ostream& oStr,
        const L1GtObject& gtObject, const int bxInEvent) const {

    bool checkBxInEvent = true;
    bool checkObjIndexInColl = false;
    int objIndexInColl = -1;

    printL1Extra(oStr, gtObject, checkBxInEvent, bxInEvent,
            checkObjIndexInColl, objIndexInColl);
}

void L1RetrieveL1Extra::printL1Extra(std::ostream& oStr,
        const L1GtObject& gtObject) const {

    bool checkBxInEvent = false;
    bool checkObjIndexInColl = false;
    int bxInEvent = 999;
    int objIndexInColl = -1;

    printL1Extra(oStr, gtObject, checkBxInEvent, bxInEvent,
            checkObjIndexInColl, objIndexInColl);
}

void L1RetrieveL1Extra::printL1Extra(std::ostream& oStr, const int iBxInEvent) const {

    printL1Extra(oStr, Mu, iBxInEvent);
    printL1Extra(oStr, NoIsoEG, iBxInEvent);
    printL1Extra(oStr, IsoEG, iBxInEvent);
    printL1Extra(oStr, CenJet, iBxInEvent);
    printL1Extra(oStr, ForJet, iBxInEvent);
    printL1Extra(oStr, TauJet, iBxInEvent);
    printL1Extra(oStr, ETM, iBxInEvent);
    printL1Extra(oStr, ETT, iBxInEvent);
    printL1Extra(oStr, HTT, iBxInEvent);
    printL1Extra(oStr, HTM, iBxInEvent);
    // printL1Extra(oStr, JetCounts, iBxInEvent);
    printL1Extra(oStr, HfBitCounts, iBxInEvent);
    printL1Extra(oStr, HfRingEtSums, iBxInEvent);
}

void L1RetrieveL1Extra::printL1Extra(std::ostream& oStr) const {

    printL1Extra(oStr, Mu);
    printL1Extra(oStr, NoIsoEG);
    printL1Extra(oStr, IsoEG);
    printL1Extra(oStr, CenJet);
    printL1Extra(oStr, ForJet);
    printL1Extra(oStr, TauJet);
    printL1Extra(oStr, ETM);
    printL1Extra(oStr, ETT);
    printL1Extra(oStr, HTT);
    printL1Extra(oStr, HTM);
    // printL1Extra(oStr, JetCounts);
    printL1Extra(oStr, HfBitCounts);
    printL1Extra(oStr, HfRingEtSums);

}
