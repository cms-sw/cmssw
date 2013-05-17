/**
 * \class L1GtTriggerMenuLiteProducer
 * 
 * 
 * Description: L1GtTriggerMenuLite producer.
 *
 * Implementation:
 *    Read the L1 trigger menu, the trigger masks and the prescale factor sets
 *    from event setup and save a lite version (top level menu, trigger masks
 *    for physics partition and prescale factor set) in Run Data.
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna 
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GtTriggerMenuLiteProducer.h"

// system include files
#include <iostream>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

// constructor(s)
L1GtTriggerMenuLiteProducer::L1GtTriggerMenuLiteProducer(
        const edm::ParameterSet& parSet) :
    m_l1GtStableParCacheID(0ULL), m_numberPhysTriggers(0),

    m_numberTechnicalTriggers(0),

    m_l1GtMenuCacheID(0ULL),

    m_l1GtTmAlgoCacheID(0ULL), m_l1GtTmTechCacheID(0ULL),

    m_l1GtPfAlgoCacheID(0ULL), m_l1GtPfTechCacheID(0ULL),

    m_physicsDaqPartition(0) {

    // EDM product in Run Data
    produces<L1GtTriggerMenuLite, edm::InRun>();

}

// destructor
L1GtTriggerMenuLiteProducer::~L1GtTriggerMenuLiteProducer() {

    // empty

}

void L1GtTriggerMenuLiteProducer::retrieveL1EventSetup(const edm::EventSetup& evSetup) {

    // get / update the stable parameters from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtStableParCacheID =
            evSetup.get<L1GtStableParametersRcd>().cacheIdentifier();

    if (m_l1GtStableParCacheID != l1GtStableParCacheID) {

        edm::ESHandle<L1GtStableParameters> l1GtStablePar;
        evSetup.get<L1GtStableParametersRcd>().get(l1GtStablePar);
        m_l1GtStablePar = l1GtStablePar.product();

        // number of physics triggers
        m_numberPhysTriggers = m_l1GtStablePar->gtNumberPhysTriggers();

        // number of technical triggers
        m_numberTechnicalTriggers =
                m_l1GtStablePar->gtNumberTechnicalTriggers();

        //
        m_l1GtStableParCacheID = l1GtStableParCacheID;

    }

    // get / update the prescale factors from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtPfAlgoCacheID =
            evSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtPfAlgoCacheID != l1GtPfAlgoCacheID) {

        edm::ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;
        evSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);
        m_l1GtPfAlgo = l1GtPfAlgo.product();

        m_prescaleFactorsAlgoTrig = &(m_l1GtPfAlgo->gtPrescaleFactors());

        m_l1GtPfAlgoCacheID = l1GtPfAlgoCacheID;

    }

    unsigned long long l1GtPfTechCacheID = evSetup.get<
            L1GtPrescaleFactorsTechTrigRcd>().cacheIdentifier();

    if (m_l1GtPfTechCacheID != l1GtPfTechCacheID) {

        edm::ESHandle<L1GtPrescaleFactors> l1GtPfTech;
        evSetup.get<L1GtPrescaleFactorsTechTrigRcd>().get(l1GtPfTech);
        m_l1GtPfTech = l1GtPfTech.product();

        m_prescaleFactorsTechTrig = &(m_l1GtPfTech->gtPrescaleFactors());

        m_l1GtPfTechCacheID = l1GtPfTechCacheID;

    }

    // get / update the trigger mask from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtTmAlgoCacheID =
            evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmAlgoCacheID != l1GtTmAlgoCacheID) {

        edm::ESHandle<L1GtTriggerMask> l1GtTmAlgo;
        evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().get(l1GtTmAlgo);
        m_l1GtTmAlgo = l1GtTmAlgo.product();

        m_triggerMaskAlgoTrig = &(m_l1GtTmAlgo->gtTriggerMask());

        m_l1GtTmAlgoCacheID = l1GtTmAlgoCacheID;

    }

    unsigned long long l1GtTmTechCacheID =
            evSetup.get<L1GtTriggerMaskTechTrigRcd>().cacheIdentifier();

    if (m_l1GtTmTechCacheID != l1GtTmTechCacheID) {

        edm::ESHandle<L1GtTriggerMask> l1GtTmTech;
        evSetup.get<L1GtTriggerMaskTechTrigRcd>().get(l1GtTmTech);
        m_l1GtTmTech = l1GtTmTech.product();

        m_triggerMaskTechTrig = &(m_l1GtTmTech->gtTriggerMask());

        m_l1GtTmTechCacheID = l1GtTmTechCacheID;

    }


    // get / update the trigger menu from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtMenuCacheID =
            evSetup.get<L1GtTriggerMenuRcd>().cacheIdentifier();

    if (m_l1GtMenuCacheID != l1GtMenuCacheID) {

        edm::ESHandle<L1GtTriggerMenu> l1GtMenu;
        evSetup.get<L1GtTriggerMenuRcd>().get(l1GtMenu);
        m_l1GtMenu = l1GtMenu.product();

        m_algorithmMap = &(m_l1GtMenu->gtAlgorithmMap());
        m_algorithmAliasMap = &(m_l1GtMenu->gtAlgorithmAliasMap());

        m_technicalTriggerMap = &(m_l1GtMenu->gtTechnicalTriggerMap());

        m_l1GtMenuCacheID = l1GtMenuCacheID;

    }

}

// member functions

void L1GtTriggerMenuLiteProducer::beginJob() {
    // empty
}

void L1GtTriggerMenuLiteProducer::beginRunProduce(edm::Run& iRun,
        const edm::EventSetup& evSetup) {

    //

    retrieveL1EventSetup(evSetup);

    // produce the L1GtTriggerMenuLite
    std::auto_ptr<L1GtTriggerMenuLite> gtTriggerMenuLite(new L1GtTriggerMenuLite());

    // lite L1 trigger menu

    gtTriggerMenuLite->setGtTriggerMenuInterface(m_l1GtMenu->gtTriggerMenuInterface());
    gtTriggerMenuLite->setGtTriggerMenuName(m_l1GtMenu->gtTriggerMenuName());
    gtTriggerMenuLite->setGtTriggerMenuImplementation(m_l1GtMenu->gtTriggerMenuImplementation());

    gtTriggerMenuLite->setGtScaleDbKey(m_l1GtMenu->gtScaleDbKey());

    //
    L1GtTriggerMenuLite::L1TriggerMap algMap;
    for (CItAlgo itAlgo = m_algorithmMap->begin(); itAlgo
            != m_algorithmMap->end(); itAlgo++) {

        unsigned int bitNumber = (itAlgo->second).algoBitNumber();
        algMap[bitNumber] = itAlgo->first;

    }

    gtTriggerMenuLite->setGtAlgorithmMap(algMap);

    //
    L1GtTriggerMenuLite::L1TriggerMap algAliasMap;
    for (CItAlgo itAlgo = m_algorithmAliasMap->begin(); itAlgo
            != m_algorithmAliasMap->end(); itAlgo++) {

        unsigned int bitNumber = (itAlgo->second).algoBitNumber();
        algAliasMap[bitNumber] = itAlgo->first;

    }

    gtTriggerMenuLite->setGtAlgorithmAliasMap(algAliasMap);

    //
    L1GtTriggerMenuLite::L1TriggerMap techMap;
    for (CItAlgo itAlgo = m_technicalTriggerMap->begin(); itAlgo
            != m_technicalTriggerMap->end(); itAlgo++) {

        unsigned int bitNumber = (itAlgo->second).algoBitNumber();
        techMap[bitNumber] = itAlgo->first;

    }

    gtTriggerMenuLite->setGtTechnicalTriggerMap(techMap);

    // trigger masks
    std::vector<unsigned int> triggerMaskAlgoTrig(m_numberPhysTriggers, 0);
    int iBit = -1;

    for (std::vector<unsigned int>::const_iterator
            itBit = m_triggerMaskAlgoTrig->begin();
            itBit != m_triggerMaskAlgoTrig->end();
            itBit++) {

        iBit++;
        triggerMaskAlgoTrig[iBit] = (*itBit) & (1 << m_physicsDaqPartition);
    }
    gtTriggerMenuLite->setGtTriggerMaskAlgoTrig(triggerMaskAlgoTrig);

    //
    std::vector<unsigned int> triggerMaskTechTrig(m_numberTechnicalTriggers, 0);
    iBit = -1;

    for (std::vector<unsigned int>::const_iterator
            itBit = m_triggerMaskTechTrig->begin();
            itBit != m_triggerMaskTechTrig->end();
            itBit++) {

        iBit++;
        triggerMaskTechTrig[iBit] = (*itBit) & (1 << m_physicsDaqPartition);
    }
    gtTriggerMenuLite->setGtTriggerMaskTechTrig(triggerMaskTechTrig);


    //
    gtTriggerMenuLite->setGtPrescaleFactorsAlgoTrig(*m_prescaleFactorsAlgoTrig);
    gtTriggerMenuLite->setGtPrescaleFactorsTechTrig(*m_prescaleFactorsTechTrig);


    // print menu, trigger masks and prescale factors
    if (edm::isDebugEnabled()) {

        LogDebug("L1GtTriggerMenuLiteProducer") << *gtTriggerMenuLite;

    }

    // put records into event
    iRun.put(gtTriggerMenuLite);

}

void L1GtTriggerMenuLiteProducer::produce(edm::Event& iEvent,
        const edm::EventSetup& evSetup) {


}

//
void L1GtTriggerMenuLiteProducer::endJob() {

    // empty now
}

// static class members

//define this as a plug-in
DEFINE_FWK_MODULE( L1GtTriggerMenuLiteProducer);
