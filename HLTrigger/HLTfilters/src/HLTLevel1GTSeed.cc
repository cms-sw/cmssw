/**
 * \class HLTLevel1GTSeed
 * 
 * 
 * Description: filter L1 bits and extract seed objects from L1 GT for HLT algorithms.  
 *
 * Implementation:
 *    This class is an HLTFilter (-> EDFilter). It implements: 
 *      - filtering on Level-1 bits, given via a logical expression of algorithm names
 *      - extraction of the seed objects from L1 GT object map record
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "HLTrigger/HLTfilters/interface/HLTLevel1GTSeed.h"

// system include files
#include <string>
#include <list>
#include <vector>
#include <algorithm>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// constructors
HLTLevel1GTSeed::HLTLevel1GTSeed(const edm::ParameterSet& parSet) {

    // logical expression for the required L1 algorithms;
    m_l1SeedsLogicalExpression = parSet.getParameter<std::string>("L1SeedsLogicalExpression");

    // InputTag for the L1 Global Trigger DAQ readout record
    m_l1GtReadoutRecordTag = parSet.getParameter<edm::InputTag>("L1GtReadoutRecordTag");

    // InputTag for L1 Global Trigger object maps
    m_l1GtObjectMapTag = parSet.getParameter<edm::InputTag>("L1GtObjectMapTag");

    // InputTag for L1 particle collections
    m_l1CollectionsTag = parSet.getParameter<edm::InputTag>("L1CollectionsTag");

    // InputTag for L1 particle collections
    m_l1MuonCollectionTag = parSet.getParameter<edm::InputTag>("L1MuonCollectionTag");

    // check logical expression - add/remove spaces if needed
    L1GtLogicParser logicParser(m_l1SeedsLogicalExpression);
    if ( !logicParser.checkLogicalExpression(m_l1SeedsLogicalExpression)) {
        
        throw cms::Exception("FailModule")
        << "\nIncorrect logical expression: " << m_l1SeedsLogicalExpression << "\n"
        << std::endl;
   
    }

    LogDebug("HLTLevel1GTSeed") << "\n" 
        << "L1 Seeds Logical Expression:        " << m_l1SeedsLogicalExpression << "\n" 
        << "Input tag for GT DAQ record:        " << m_l1GtReadoutRecordTag.label() << " \n" 
        << "Input tag for GT object map record: " << m_l1GtObjectMapTag.label() << " \n" 
        << "Input tag for L1 extra collections: " << m_l1CollectionsTag.label() << " \n" 
        << "Input tag for L1 muon  collections: " << m_l1MuonCollectionTag.label() << " \n" 
        << std::endl;

    // register the products
    produces<trigger::TriggerFilterObjectWithRefs>();
}

// destructor
HLTLevel1GTSeed::~HLTLevel1GTSeed() {
    // empty now
}

// member functions

bool HLTLevel1GTSeed::filter(edm::Event& iEvent, const edm::EventSetup& evSetup) {

    // all HLT filters must create and fill a HLT filter object,
    // recording any reconstructed physics objects satisfying (or not) // TODO "or not" ??
    // this HLT filter, and place it in the event.

    // the filter object
    std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterObject (
        new trigger::TriggerFilterObjectWithRefs( path(), module() ) );

    // get L1GlobalTriggerReadoutRecord and GT decision
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_l1GtReadoutRecordTag, gtReadoutRecord);

    bool gtDecision = gtReadoutRecord->decision();

    // GT global decision "false" possible only when running on MC or on random triggers
    if ( !gtDecision) {

        iEvent.put(filterObject);
        return false;

    }
    else {

        // by convention, "L1GlobalDecision" logical expression means global decision
        if (m_l1SeedsLogicalExpression == "L1GlobalDecision") {

            // return the full L1GlobalTriggerObjectMapRecord in filter format FIXME
            iEvent.put(filterObject);

            return true;

        }

    }

    
    // get the trigger menu from the EventSetup
    edm::ESHandle< L1GtTriggerMenu> l1GtMenu;
    evSetup.get< L1GtTriggerMenuRcd>().get(l1GtMenu) ;

    const AlgorithmMap algorithmMap = l1GtMenu->gtAlgorithmMap();
    const std::vector<ConditionMap> conditionMap = l1GtMenu->gtConditionMap();

    // get Global Trigger decision word and the result for the logical expression
    DecisionWord gtDecisionWord = gtReadoutRecord->decisionWord();

    std::map<std::string, int> algNameToBit = mapAlgNameToBit(algorithmMap);
    
    L1GtLogicParser logicParserAlgorithms(m_l1SeedsLogicalExpression, gtDecisionWord, algNameToBit);
    bool seedsResult = logicParserAlgorithms.expressionResult();

    if (edm::isDebugEnabled() ) {
        // define an output stream to print into
        // it can then be directed to whatever log level is desired
        std::ostringstream myCoutStream;
        gtReadoutRecord->printGtDecision(myCoutStream);

        LogTrace("HLTLevel1GTSeed")
            << myCoutStream.str()
            << "\nHLTLevel1GTSeed::filter "
            << "\nLogical expression (names) = '" << m_l1SeedsLogicalExpression << "'"
            << "\n  Result for logical expression: " << seedsResult << "\n"
            << std::endl;
    }

    // the evaluation of the logical expression is false - skip event
    if ( !seedsResult) {

        iEvent.put(filterObject);
        return false;

    }

    // define index lists for all particle types

    std::list<int> listMuon;

    std::list<int> listIsoEG;
    std::list<int> listNoIsoEG;

    std::list<int> listCenJet;
    std::list<int> listForJet;
    std::list<int> listTauJet;

    std::list<int> listETM;
    std::list<int> listETT;
    std::list<int> listHTT;

    std::list<int> listJetCounts;

    // get handle to object maps (one object map per algorithm)
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByLabel(m_l1GtObjectMapTag, gtObjectMapRecord);

    // get list of required algorithms for seeding - loop over
    std::vector<L1GtLogicParser::OperandToken> algSeeds =
        logicParserAlgorithms.expressionSeedsOperandList();

    for (std::vector<L1GtLogicParser::OperandToken>::const_iterator 
        itSeed = algSeeds.begin(); itSeed != algSeeds.end(); ++itSeed) {

        int algBit = (*itSeed).tokenNumber;
        std::string algName = (*itSeed).tokenName;
        bool algResult = (*itSeed).tokenResult;

        LogTrace("HLTLevel1GTSeed")
            << "\nHLTLevel1GTSeed::filter "
            << "\n  Algoritm " << algName << " with bit number " << algBit 
            << " in the object map seed list"
            << "\n  Algorithm result = " << algResult << "\n"
            << std::endl;


        // algorithm result is false - no seeds
        if ( !algResult) {
            continue;
        }

        // algorithm result is true - get object map, loop over conditions in the algorithm
        const L1GlobalTriggerObjectMap* objMap = gtObjectMapRecord->getObjectMap(algName);

        L1GtLogicParser logicParserConditions( (*objMap));

        // get list of required conditions for seeding - loop over
        std::vector<L1GtLogicParser::OperandToken> condSeeds =
            logicParserConditions.expressionSeedsOperandList();

        for (std::vector<L1GtLogicParser::OperandToken>::const_iterator 
            itCond = condSeeds.begin(); itCond != condSeeds.end(); itCond++) {

            std::string cndName = (*itCond).tokenName;
            bool cndResult = (*itCond).tokenResult;

            if ( !cndResult) {
                continue;
            }

            // loop over combinations for a given condition
            
            const CombinationsInCond* cndComb = objMap->getCombinationsInCond(cndName);

            for (std::vector<SingleCombInCond>::const_iterator 
                itComb = (*cndComb).begin(); itComb != (*cndComb).end(); itComb++) {

                // loop over objects in a combination for a given condition
                int iObj = 0;
                for (SingleCombInCond::const_iterator 
                    itObject = (*itComb).begin(); itObject != (*itComb).end(); itObject++) {

                    // get object type and push indices on the list
                    const L1GtObject objTypeVal = objectType(cndName, iObj, conditionMap);

                    LogTrace("HLTLevel1GTSeed")
                        << "\nHLTLevel1GTSeed::filter "
                        << "\n  Add object of type " << objTypeVal
                        << " and index " << (*itObject) << " to the seed list."
                        << std::endl;

                    switch (objTypeVal) {
                        case Mu: {
                                listMuon.push_back(*itObject);
                            }

                            break;
                        case NoIsoEG: {
                                listNoIsoEG.push_back(*itObject);
                            }

                            break;
                        case IsoEG: {
                                listIsoEG.push_back(*itObject);
                            }

                            break;
                        case CenJet: {
                                listCenJet.push_back(*itObject);
                            }

                            break;
                        case ForJet: {
                                listForJet.push_back(*itObject);
                            }

                            break;
                        case TauJet: {
                                listTauJet.push_back(*itObject);
                            }

                            break;
                        case ETM: {
                                listETM.push_back(*itObject);

                            }

                            break;
                        case ETT: {
                                listETT.push_back(*itObject);

                            }

                            break;
                        case HTT: {
                                listHTT.push_back(*itObject);

                            }

                            break;
                        case JetCounts: {
                                listJetCounts.push_back(*itObject);
                            }

                            break;
                        default: {
                                // should not arrive here
                            }
                            break;
                    }

                    iObj++;

                }

            }


        }

    }

    // eliminate duplicates

    listMuon.sort();
    listMuon.unique();

    listIsoEG.sort();
    listIsoEG.unique();

    listNoIsoEG.sort();
    listNoIsoEG.unique();

    listCenJet.sort();
    listCenJet.unique();

    listForJet.sort();
    listForJet.unique();

    listTauJet.sort();
    listTauJet.unique();

    // no need to eliminate duplicates for energy sums and jet counts
    // they are global quantities


    //
    // record the L1 physics objects in the HLT filterObject
    //

    // muon
    if (listMuon.size()) {

        edm::Handle<l1extra::L1MuonParticleCollection> l1Muon;
        edm::InputTag l1MuonTag(edm::InputTag(m_l1MuonCollectionTag.label()) );
        iEvent.getByLabel(l1MuonTag, l1Muon);

        for (std::list<int>::const_iterator itObj = listMuon.begin(); itObj != listMuon.end(); ++itObj) {

	    filterObject->addObject(trigger::TriggerL1Mu,l1extra::L1MuonParticleRef(l1Muon, *itObj));

        }
    }


    // EG (isolated)
    if (listIsoEG.size()) {
        edm::Handle<l1extra::L1EmParticleCollection> l1IsoEG;
        edm::InputTag l1IsoEGTag( edm::InputTag(m_l1CollectionsTag.label(), "Isolated") );
        iEvent.getByLabel(l1IsoEGTag, l1IsoEG);

        for (std::list<int>::const_iterator 
            itObj = listIsoEG.begin(); itObj != listIsoEG.end(); ++itObj) {

	    filterObject->addObject(trigger::TriggerL1IsoEG,l1extra::L1EmParticleRef(l1IsoEG, *itObj));

        }
    }

    // EG (no isolation)
    if (listNoIsoEG.size()) {
        edm::Handle<l1extra::L1EmParticleCollection> l1NoIsoEG;
        edm::InputTag l1NoIsoEGTag( edm::InputTag(m_l1CollectionsTag.label(), "NonIsolated") );
        iEvent.getByLabel(l1NoIsoEGTag, l1NoIsoEG);

        for (std::list<int>::const_iterator 
            itObj = listNoIsoEG.begin(); itObj != listNoIsoEG.end(); ++itObj) {

	    filterObject->addObject(trigger::TriggerL1NoIsoEG,l1extra::L1EmParticleRef(l1NoIsoEG, *itObj));

        }
    }

    // central jets
    if (listCenJet.size()) {
        edm::Handle<l1extra::L1JetParticleCollection> l1CenJet;
        edm::InputTag l1CenJetTag( edm::InputTag(m_l1CollectionsTag.label(), "Central") );
        iEvent.getByLabel(l1CenJetTag, l1CenJet);

        for (std::list<int>::const_iterator 
            itObj = listCenJet.begin(); itObj != listCenJet.end(); ++itObj) {

	    filterObject->addObject(trigger::TriggerL1CenJet,l1extra::L1JetParticleRef(l1CenJet, *itObj));

        }
    }

    // forward jets
    if (listForJet.size()) {
        edm::Handle<l1extra::L1JetParticleCollection> l1ForJet;
        edm::InputTag l1ForJetTag( edm::InputTag(m_l1CollectionsTag.label(), "Forward") );
        iEvent.getByLabel(l1ForJetTag, l1ForJet);

        for (std::list<int>::const_iterator 
            itObj = listForJet.begin(); itObj != listForJet.end(); ++itObj) {

	    filterObject->addObject(trigger::TriggerL1ForJet,l1extra::L1JetParticleRef(l1ForJet, *itObj));

        }
    }

    // tau jets
    if (listTauJet.size()) {
        edm::Handle<l1extra::L1JetParticleCollection> l1TauJet;
        edm::InputTag l1TauJetTag( edm::InputTag(m_l1CollectionsTag.label(), "Tau") );
        iEvent.getByLabel(l1TauJetTag, l1TauJet);

        for (std::list<int>::const_iterator itObj = listTauJet.begin();
            itObj != listTauJet.end(); ++itObj) {

	    filterObject->addObject(trigger::TriggerL1TauJet,l1extra::L1JetParticleRef(l1TauJet, *itObj));

        }
    }

    // energy sums
    if (listETM.size() || listETT.size() || listHTT.size()) {
        edm::Handle<l1extra::L1EtMissParticleCollection> l1EnergySums;
        iEvent.getByLabel(m_l1CollectionsTag.label(), l1EnergySums);

        for (std::list<int>::const_iterator 
            itObj = listETM.begin(); itObj != listETM.end(); ++itObj) {

	    filterObject->addObject(trigger::TriggerL1ETM,l1extra::L1EtMissParticleRef(l1EnergySums, *itObj));

        }

        for (std::list<int>::const_iterator 
            itObj = listETT.begin(); itObj != listETT.end(); ++itObj) {

	    filterObject->addObject(trigger::TriggerL1ETT,l1extra::L1EtMissParticleRef(l1EnergySums, *itObj));

        }

        for (std::list<int>::const_iterator 
            itObj = listHTT.begin(); itObj != listHTT.end(); ++itObj) {

	    filterObject->addObject(trigger::TriggerL1HTT,l1extra::L1EtMissParticleRef(l1EnergySums, *itObj));

        }

    }


    // TODO FIXME uncomment if block when JetCounts implemented

    //    // jet counts
    //    if (listJetCounts.size()) {
    //        edm::Handle<l1extra::L1JetCounts> l1JetCounts;
    //        iEvent.getByLabel(m_l1CollectionsTag.label(), l1JetCounts);
    //
    //        for (std::list<int>::const_iterator itObj = listJetCounts.begin();
    //                itObj != listJetCounts.end(); ++itObj) {
    //
    //            filterObject->addObject(trigger::TriggerL1JetCounts,l1extra::L1JetCountsRefProd(l1JetCounts));
    //	          // FIXME: RefProd!
    //
    //        }
    //
    //    }

    /* FIXME: must be updated to new HLT data model

    if ( edm::isDebugEnabled() ) {

        LogDebug("HLTLevel1GTSeed")
            << "\nHLTLevel1GTSeed::filter "
            << "\n  Dump HLTFilterObjectWithRefs\n"
            << std::endl;

        const unsigned int n = filterObjectOLD->size();
        LogTrace("HLTLevel1GTSeed") << "  Size = " << n;
        for (unsigned int i = 0; i != n; i++) {

            reco::Particle particle = filterObjectOLD->getParticle(i);
            const reco::Candidate* candidate = (filterObjectOLD->getParticleRef(i)).get();

            LogTrace("HLTLevel1GTSeed")
                << "   " << i << "\t"
                << typeid(*candidate).name() << "\t"
                << "E = " << particle.energy() << " = " << candidate->energy() << "\t"
                << "eta = " << particle.eta() << "\t"
                << "phi = " << particle.phi();
        }

        LogTrace("HLTLevel1GTSeed") << " \n\n"
        << std::endl;

    }
    */ // end FIXME

    iEvent.put(filterObject);

    return seedsResult;

}

L1GtObject HLTLevel1GTSeed::objectType(const std::string& cndName, const int& indexObj,
    const std::vector<ConditionMap>& conditionMap) {

    bool foundCond = false;
    
    for (std::vector<ConditionMap>::const_iterator 
        itVec = conditionMap.begin(); itVec != conditionMap.end(); itVec++) {

        CItCond itCond = (*itVec).find(cndName);
        if (itCond != (*itVec).end()) {
            foundCond = true;
            return ((*(itCond->second)).objectType())[indexObj];
        }
    }
    
    if ( !foundCond) {

        // it should never be happen, all conditions are in the maps
        throw cms::Exception("FailModule")
        << "\nCondition " << cndName << " not found in condition map"
        << std::endl;
    }

    // dummy return - prevent compiler warning
    return Mu;

}

// get map of (algorithm names, algorithm bits)
std::map<std::string, int> HLTLevel1GTSeed::mapAlgNameToBit(const AlgorithmMap& algorithmMap) {

    std::map<std::string, int> algNameToBit;

    for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {

        std::string algName = itAlgo->first;
        int algBitNumber = (itAlgo->second)->algoBitNumber();

        algNameToBit[algName] = algBitNumber;

    }

    return algNameToBit;
}

