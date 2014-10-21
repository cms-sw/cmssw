/**
 * \class L1GlobalTriggerRecordProducer
 * 
 * 
 * Description: L1GlobalTriggerRecord producer.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna 
 * 
 *
 */


// this class header
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerRecordProducer.h"

// system include files
#include <iostream>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

// constructor(s)
L1GlobalTriggerRecordProducer::L1GlobalTriggerRecordProducer(const edm::ParameterSet& parSet)
{

    produces<L1GlobalTriggerRecord>();

    // input tag for DAQ GT record
    m_l1GtReadoutRecordTag = 
        parSet.getParameter<edm::InputTag>("L1GtReadoutRecordTag");

    LogDebug("L1GlobalTriggerRecordProducer")
    << "\nInput tag for L1 GT DAQ record:             "
    << m_l1GtReadoutRecordTag
    << std::endl;

    // initialize cached IDs
    
    m_l1GtTmAlgoCacheID = 0ULL;
    m_l1GtTmTechCacheID = 0ULL;
    
}

// destructor
L1GlobalTriggerRecordProducer::~L1GlobalTriggerRecordProducer()
{

    // empty
    
}


// member functions

// method called to produce the data
void L1GlobalTriggerRecordProducer::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // produce the L1GlobalTriggerRecord
    std::auto_ptr<L1GlobalTriggerRecord> gtRecord(new L1GlobalTriggerRecord());

    // get L1GlobalTriggerReadoutRecord
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_l1GtReadoutRecordTag, gtReadoutRecord);
    
    if (!gtReadoutRecord.isValid()) {
        
        LogDebug("L1GlobalTriggerRecordProducer")
        << "\n\n Error: no L1GlobalTriggerReadoutRecord found with input tag "
        << m_l1GtReadoutRecordTag
        << "\n Returning empty L1GlobalTriggerRecord.\n\n"
        << std::endl;
        
        iEvent.put( gtRecord );
        return;
    }

    //
    cms_uint16_t gtFinalOR = gtReadoutRecord->finalOR();
    int physicsDaqPartition = 0;
    bool gtDecision = static_cast<bool> (gtFinalOR & (1 << physicsDaqPartition));
    
    DecisionWord algoDecisionWord = gtReadoutRecord->decisionWord();    
    TechnicalTriggerWord techDecisionWord = gtReadoutRecord->technicalTriggerWord();   

    // get / update the trigger mask from the EventSetup 
    // local cache & check on cacheIdentifier

    unsigned long long l1GtTmAlgoCacheID = 
        evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmAlgoCacheID != l1GtTmAlgoCacheID) {
        
        edm::ESHandle< L1GtTriggerMask > l1GtTmAlgo;
        evSetup.get< L1GtTriggerMaskAlgoTrigRcd >().get( l1GtTmAlgo );        
        m_l1GtTmAlgo = l1GtTmAlgo.product();
        
        m_triggerMaskAlgoTrig = m_l1GtTmAlgo->gtTriggerMask();
        
        m_l1GtTmAlgoCacheID = l1GtTmAlgoCacheID;

    }
    

    unsigned long long l1GtTmTechCacheID = 
        evSetup.get<L1GtTriggerMaskTechTrigRcd>().cacheIdentifier();

    if (m_l1GtTmTechCacheID != l1GtTmTechCacheID) {
        
        edm::ESHandle< L1GtTriggerMask > l1GtTmTech;
        evSetup.get< L1GtTriggerMaskTechTrigRcd >().get( l1GtTmTech );        
        m_l1GtTmTech = l1GtTmTech.product();
        
        m_triggerMaskTechTrig = m_l1GtTmTech->gtTriggerMask();
        
        m_l1GtTmTechCacheID = l1GtTmTechCacheID;

    }
    
    /// set global decision, decision word and technical trigger word
    /// for bunch cross with L1Accept (BxInEvent = 0) before applying the masks
    gtRecord->setDecisionWordBeforeMask(algoDecisionWord);
    gtRecord->setTechnicalTriggerWordBeforeMask(techDecisionWord);

    // mask the required bits for DAQ partition 0 (Physics Partition)
    
    int iDaq = 0;
    
    // algorithm trigger mask
    
    int iBit = -1; // bit counter

    for (std::vector<bool>::iterator 
            itBit = algoDecisionWord.begin(); itBit != algoDecisionWord.end(); ++itBit) {
        
        iBit++;
        
        int triggerMaskAlgoTrigBit = m_triggerMaskAlgoTrig[iBit] & (1 << iDaq);
        //LogTrace("L1GlobalTriggerFDL")
        //<< "\nAlgorithm trigger bit: " << iBit 
        //<< " mask = " << triggerMaskAlgoTrigBit
        //<< " DAQ partition " << iDaq
        //<< std::endl;

        if (triggerMaskAlgoTrigBit) {
            *itBit = false;

            //LogTrace("L1GlobalTriggerFDL")
            //<< "\nMasked algorithm trigger: " << iBit << ". Result set to false"
            //<< std::endl;
        }
    }

    // mask the technical trigger

    iBit = -1; // bit counter

    for (std::vector<bool>::iterator 
            itBit = techDecisionWord.begin(); itBit != techDecisionWord.end(); ++itBit) {
        
        iBit++;
        
        int triggerMaskTechTrigBit = m_triggerMaskTechTrig[iBit] & (1 << iDaq);
        //LogTrace("L1GlobalTriggerFDL")
        //<< "\nTechnical trigger bit: " << iBit 
        //<< " mask = " << triggerMaskTechTrigBit
        //<< " DAQ partition " << iDaq
        //<< std::endl;

        if (triggerMaskTechTrigBit) {
            *itBit = false;

            //LogTrace("L1GlobalTriggerFDL")
            //<< "\nMasked technical trigger: " << iBit << ". Result set to false"
            //<< std::endl;
        }
    }

    
    

    // set global decision, decision word and technical trigger word
    // for bunch cross with L1Accept (BxInEvent = 0) after applying the trigger masks
    gtRecord->setDecision(gtDecision);
    gtRecord->setDecisionWord(algoDecisionWord);
    gtRecord->setTechnicalTriggerWord(techDecisionWord);
    
    // get/set index of the set of prescale factors
    unsigned int pfIndexTech = 
        static_cast<unsigned int> ((gtReadoutRecord->gtFdlWord()).gtPrescaleFactorIndexTech());
    unsigned int pfIndexAlgo =
        static_cast<unsigned int> ((gtReadoutRecord->gtFdlWord()).gtPrescaleFactorIndexAlgo());
    
    gtRecord->setGtPrescaleFactorIndexTech(pfIndexTech);
    gtRecord->setGtPrescaleFactorIndexAlgo(pfIndexAlgo);
    
    if ( edm::isDebugEnabled() ) {
        std::ostringstream myCoutStream;
        gtRecord->print(myCoutStream);
        LogTrace("L1GlobalTriggerRecordProducer")
        << "\n The following L1 GT record was produced.\n"
        << myCoutStream.str() << "\n"
        << std::endl;
    }

    // put records into event
    iEvent.put( gtRecord );

}

// static class members

