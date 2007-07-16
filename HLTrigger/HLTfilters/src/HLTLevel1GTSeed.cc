/**
 * \class HLTLevel1GTSeed
 * 
 * 
 * Description: see header file for documentation.  
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "HLTrigger/HLTfilters/interface/HLTLevel1GTSeed.h"

// system include files
#include <string>
#include <list>
#include <algorithm>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"

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

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// constructors
HLTLevel1GTSeed::HLTLevel1GTSeed(const edm::ParameterSet& parSet)
{


    // logical expression for the required L1 algorithms;
    m_l1SeedsLogicalExpression = parSet.getParameter<std::string> ("L1SeedsLogicalExpression");

    // by convention, "L1GlobalDecision" logical expression means global decision
    // TODO FIXME talk with Martin about convention, build the expression
    if (m_l1SeedsLogicalExpression == "L1GlobalDecision") {

        // build a general OR and return the full L1GlobalTriggerObjectMapRecord

    }

    // InputTag for the L1 Global Trigger DAQ readout record
    m_l1GtReadoutRecordTag = parSet.getParameter<edm::InputTag> ("L1GtReadoutRecordTag");

    // InputTag for L1 Global Trigger object maps
    m_l1GtObjectMapTag = parSet.getParameter<edm::InputTag> ("L1GtObjectMapTag");

    /// InputTag for L1 particle collections
    m_l1CollectionsTag = parSet.getParameter<edm::InputTag> ("L1CollectionsTag");

    // TODO FIXME temporary, until L1 trigger menu implemented as EventSetup
    m_algoMap = false;

    LogDebug("HLTLevel1GTSeed")
    << "\n"
    << "L1 Seeds Logical Expression:        " << m_l1SeedsLogicalExpression << "\n"
    << "Input tag for GT DAQ record:        " << m_l1GtReadoutRecordTag.label() << " \n"
    << "Input tag for GT object map record: " << m_l1GtObjectMapTag.label() << " \n"
    << "Input tag for L1 extra collections: " << m_l1CollectionsTag.label() << " \n"
    << std::endl;

    // register the products
    produces<reco::HLTFilterObjectWithRefs>();
}

// destructor
HLTLevel1GTSeed::~HLTLevel1GTSeed()
{
    // empty now
}

// member functions

bool HLTLevel1GTSeed::filter(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // all HLT filters must create and fill a HLT filter object,
    // recording any reconstructed physics objects satisfying (or not) // TODO "or not" ??
    // this HLT filter, and place it in the event.

    // the filter object
    std::auto_ptr<reco::HLTFilterObjectWithRefs> filterObject (
        new reco::HLTFilterObjectWithRefs( path(), module() ) );


    // get L1GlobalTriggerReadoutRecord and GT decision
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_l1GtReadoutRecordTag.label(), gtReadoutRecord);

    bool gtDecision = gtReadoutRecord->decision();

    // GT global decision "false" possible only when running on MC or on random triggers
    if ( !gtDecision) {

        iEvent.put(filterObject);
        return gtDecision;

    }

    // TODO FIXME temporary
    if ( !m_algoMap ) {

        // get map from algorithm names to algorithm bits
        getAlgoMap(iEvent, evSetup);

        // convert m_l1SeedsLogicalExpression from algorithm names to algorithm bits
        // it should be faster that with trigger names
        L1GtLogicParser logicParser(m_l1SeedsLogicalExpression);

        logicParser.convertNameToIntLogicalExpression(m_algoNameToBit);
        m_l1SeedsLogicalExpression = logicParser.logicalExpression();

        m_algoMap = true;

    }

    // get Global Trigger decision word and the result for the logical expression
    DecisionWord gtDecisionWord = gtReadoutRecord->decisionWord();

    L1GtLogicParser logicParser(m_l1SeedsLogicalExpression, gtDecisionWord, m_algoNameToBit);
    bool seedsResult = logicParser.expressionResult();

    if ( edm::isDebugEnabled() ) {
        // define an output stream to print into
        // it can then be directed to whatever log level is desired
        std::ostringstream myCoutStream;
        gtReadoutRecord->printGtDecision(myCoutStream);

        LogTrace("HLTLevel1GTSeed")
        << myCoutStream.str()
        << "\nHLTLevel1GTSeed::filter "
        << "\nLogical expression = '" << m_l1SeedsLogicalExpression << "'"
        << "\n  Result for logical expression: " << seedsResult << "\n"
        << std::endl;
    }

    // the evaluation of the logical expression is false - skip event
    if ( !seedsResult) {

        iEvent.put(filterObject);
        return seedsResult;

    }

    // get list of required object maps (list of algorithm bit numbers)
    // using the logical expression
    std::list<int> objectMapList = logicParser.expressionSeedsOperandList();

    // get object maps (one object map per algorithm)
    // the record can come only from emulator - no hardware ObjectMapRecord
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByLabel(m_l1GtObjectMapTag.label(), gtObjectMapRecord);

    const std::vector<L1GlobalTriggerObjectMap>& objMapVec =
        gtObjectMapRecord->gtObjectMap();

    LogDebug("HLTLevel1GTSeed")
    << "\nHLTLevel1GTSeed::filter "
    << "\n  Size of object map vector = " << objMapVec.size() << "\n"
    << std::endl;

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

    // loop over algorithms (one object map per algorithm)

    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
            itMap != objMapVec.end(); ++itMap) {

        // check if the map is needed (using algorithm bits)
        int algoBit = (*itMap).algoBitNumber();
        if (std::count( objectMapList.begin(), objectMapList.end(), algoBit ) == 0) {

            continue;
        }

        int algoResult = gtDecisionWord.at(algoBit);

        LogTrace("HLTLevel1GTSeed")
        << "\nHLTLevel1GTSeed::filter "
        << "\n  Algoritm with bit number " << algoBit
        << " in the object map seed list"
        << "\n  Algorithm result = " << algoResult << "\n"
        << std::endl;


        // algorithm result is false - no seeds
        if ( !algoResult ) {

            continue;

        }

        // loop over conditions in an algorithm

        L1GtLogicParser logicParserConditions( (*itMap) );

        // conditions required (list of indices)
        std::list<int> condList =
            logicParserConditions.expressionSeedsOperandIndexList();

        const std::vector<CombinationsInCond> combVector = itMap->combinationVector();
        const std::vector<ObjectTypeInCond> typeInCond =  itMap->objectTypeVector(); // TODO
        int iCond = 0;

        for(std::vector<CombinationsInCond>::const_iterator itCond = combVector.begin();
                itCond != combVector.end(); itCond++) {

            // condition not in the list
            if (std::count( condList.begin(), condList.end(), iCond ) == 0) {
                
                LogTrace("HLTLevel1GTSeed")
                << "\nHLTLevel1GTSeed::filter "
                << "\n  Condition index " << iCond
                << " not in seed condition list."
                << std::endl;

                iCond++;
                continue;
            }

            // condition in the list
            bool condResult = logicParserConditions.operandResult(iCond);

            if ( !condResult ) {
                iCond++;
                continue;
            }

            // loop over combinations for a given condition
            for(std::vector<SingleCombInCond>::const_iterator itComb = (*itCond).begin();
                    itComb != (*itCond).end(); itComb++) {

                const ObjectTypeInCond objTypeT = typeInCond[iCond]; // TODO

                // loop over objects in a combination for a given condition
                int iObj = 0;
                for(SingleCombInCond::const_iterator itObject = (*itComb).begin();
                        itObject != (*itComb).end(); itObject++) {

                    // get object type and push indices on the list
                    // TODO adapt to EventSetup, whe available
                    // const L1GtObject objTypeVal = objectType(algoBit, iCond, iObj, evSetup);
                    const L1GtObject objTypeVal = objTypeT[iObj];

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

                }

                iObj++;
            }

        }

        iCond++;
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

    // ref to Candidate object to be recorded in filter object
    edm::RefToBase<reco::Candidate> ref;

    // muon
    if (listMuon.size()) {

        edm::Handle<l1extra::L1MuonParticleCollection> l1Muon;
        edm::InputTag l1MuonTag( edm::InputTag(m_l1CollectionsTag.label()) );
        iEvent.getByLabel(l1MuonTag, l1Muon);

        for (std::list<int>::const_iterator itObj = listMuon.begin();
                itObj != listMuon.end(); ++itObj) {

            ref = edm::RefToBase<reco::Candidate>(l1extra::L1MuonParticleRef(l1Muon, *itObj));
            filterObject->putParticle(ref);

        }
    }


    // EG (isolated)
    if (listIsoEG.size()) {
        edm::Handle<l1extra::L1EmParticleCollection> l1IsoEG;
        edm::InputTag l1IsoEGTag( edm::InputTag(m_l1CollectionsTag.label(),
                                                "Isolated") );
        iEvent.getByLabel(l1IsoEGTag, l1IsoEG);

        for (std::list<int>::const_iterator itObj = listIsoEG.begin();
                itObj != listIsoEG.end(); ++itObj) {

            ref = edm::RefToBase<reco::Candidate>(l1extra::L1EmParticleRef(l1IsoEG, *itObj));
            filterObject->putParticle(ref);

        }
    }

    // EG (no isolation)
    if (listNoIsoEG.size()) {
        edm::Handle<l1extra::L1EmParticleCollection> l1NoIsoEG;
        edm::InputTag l1NoIsoEGTag( edm::InputTag(m_l1CollectionsTag.label(),
                                    "NonIsolated") );
        iEvent.getByLabel(l1NoIsoEGTag, l1NoIsoEG);

        for (std::list<int>::const_iterator itObj = listNoIsoEG.begin();
                itObj != listNoIsoEG.end(); ++itObj) {

            ref = edm::RefToBase<reco::Candidate>(l1extra::L1EmParticleRef(l1NoIsoEG, *itObj));
            filterObject->putParticle(ref);

        }
    }

    // central jets
    if (listCenJet.size()) {
        edm::Handle<l1extra::L1JetParticleCollection> l1CenJet;
        edm::InputTag l1CenJetTag( edm::InputTag(m_l1CollectionsTag.label(),
                                   "Central") );
        iEvent.getByLabel(l1CenJetTag, l1CenJet);

        for (std::list<int>::const_iterator itObj = listCenJet.begin();
                itObj != listCenJet.end(); ++itObj) {

            ref = edm::RefToBase<reco::Candidate>(l1extra::L1JetParticleRef(l1CenJet, *itObj));
            filterObject->putParticle(ref);

        }
    }

    // forward jets
    if (listForJet.size()) {
        edm::Handle<l1extra::L1JetParticleCollection> l1ForJet;
        edm::InputTag l1ForJetTag( edm::InputTag(m_l1CollectionsTag.label(),
                                   "Forward") );
        iEvent.getByLabel(l1ForJetTag, l1ForJet);

        for (std::list<int>::const_iterator itObj = listForJet.begin();
                itObj != listForJet.end(); ++itObj) {

            ref = edm::RefToBase<reco::Candidate>(l1extra::L1JetParticleRef(l1ForJet, *itObj));
            filterObject->putParticle(ref);

        }
    }

    // tau jets
    if (listTauJet.size()) {
        edm::Handle<l1extra::L1JetParticleCollection> l1TauJet;
        edm::InputTag l1TauJetTag( edm::InputTag(m_l1CollectionsTag.label(),
                                   "Tau") );
        iEvent.getByLabel(l1TauJetTag, l1TauJet);

        for (std::list<int>::const_iterator itObj = listTauJet.begin();
                itObj != listTauJet.end(); ++itObj) {

            ref = edm::RefToBase<reco::Candidate>(l1extra::L1JetParticleRef(l1TauJet, *itObj));
            filterObject->putParticle(ref);

        }
    }

    // energy sums
    if (listETM.size() || listETT.size() || listHTT.size()) {
        edm::Handle<l1extra::L1EtMissParticle> l1EnergySums;
        iEvent.getByLabel(m_l1CollectionsTag.label(), l1EnergySums);

        for (std::list<int>::const_iterator itObj = listETM.begin();
                itObj != listETM.end(); ++itObj) {

            ref = edm::RefToBase<reco::Candidate>(l1extra::L1EtMissParticleRefProd(l1EnergySums));
            filterObject->putParticle(ref);

        }

        for (std::list<int>::const_iterator itObj = listETT.begin();
                itObj != listETT.end(); ++itObj) {

            ref = edm::RefToBase<reco::Candidate>(l1extra::L1EtMissParticleRefProd(l1EnergySums));
            filterObject->putParticle(ref);

        }

        for (std::list<int>::const_iterator itObj = listHTT.begin();
                itObj != listHTT.end(); ++itObj) {

            ref = edm::RefToBase<reco::Candidate>(l1extra::L1EtMissParticleRefProd(l1EnergySums));
            filterObject->putParticle(ref);

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
    //            ref = edm::RefToBase<reco::Candidate>(l1extra::L1JetCountsRefProd(l1JetCounts));
    //            filterObject->putParticle(ref);
    //
    //        }
    //
    //    }

    if ( edm::isDebugEnabled() ) {

        LogDebug("HLTLevel1GTSeed")
        << "\nHLTLevel1GTSeed::filter "
        << "\n  Dump HLTFilterObjectWithRefs\n"
        << std::endl;

        const unsigned int n = filterObject->size();
        LogTrace("HLTLevel1GTSeed") << "  Size = " << n;
        for (unsigned int i = 0; i != n; i++) {

            reco::Particle particle  = filterObject->getParticle(i);
            const reco::Candidate* candidate = (filterObject->getParticleRef(i)).get();

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

    iEvent.put(filterObject);
    return seedsResult;

}

L1GtObject HLTLevel1GTSeed::objectType(
    const int algoBit, const int indexCond, const int indexObj,
    const edm::EventSetup& evSetup)
{
    // TODO FIXME dummy
    // TODO get the object type from EventSetup trigger menu

    L1GtObject objType = Mu;

    return objType;

}

// TODO FIXME temporary solution, until L1 trigger menu is implemented as event setup
// get map from algorithm names to algorithm bits
void HLTLevel1GTSeed::getAlgoMap(
    edm::Event& iEvent, const edm::EventSetup& evSetup)
{


    // get object maps (one object map per algorithm)
    // the record can come only from emulator - no hardware ObjectMapRecord
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByLabel(m_l1GtObjectMapTag.label(), gtObjectMapRecord);

    const std::vector<L1GlobalTriggerObjectMap>& objMapVec =
        gtObjectMapRecord->gtObjectMap();

    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
            itMap != objMapVec.end(); ++itMap) {

        int algoBit = (*itMap).algoBitNumber();
        std::string algoNameStr = (*itMap).algoName();

        m_algoNameToBit[algoNameStr] = algoBit;

    }

    if ( edm::isDebugEnabled() ) {

        LogDebug("HLTLevel1GTSeed")
        << "\nHLTLevel1GTSeed::getAlgoMap "
        << "\n  L1 Trigger menu: map for algorithm name and bits\n"
        << std::endl;

        typedef std::map<std::string, int>::const_iterator CIter;

        for (CIter it = m_algoNameToBit.begin(); it != m_algoNameToBit.end(); ++it) {

            LogTrace("HLTLevel1GTSeed")
            << it->first << "  " <<  it->second
            << std::endl;
        }
    }

}
