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
 * $Date$
 * $Revision$
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
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


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

}

// destructor
L1GlobalTriggerRecordProducer::~L1GlobalTriggerRecordProducer()
{

    // empty
    
}


// member functions

void L1GlobalTriggerRecordProducer::beginJob(const edm::EventSetup& evSetup)
{
    // empty
}

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
    const bool gtDecision = gtReadoutRecord->decision();
    const DecisionWord gtDecisionWord = gtReadoutRecord->decisionWord();    
    const TechnicalTriggerWord ttDecisionWord = gtReadoutRecord->technicalTriggerWord();   

    /// set global decision, decision word and technical trigger word
    /// for bunch cross with L1Accept (BxInEvent = 0) 
    gtRecord->setDecision(gtDecision);
    gtRecord->setDecisionWord(gtDecisionWord);
    gtRecord->setTechnicalTriggerWord(ttDecisionWord);

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

//
void L1GlobalTriggerRecordProducer::endJob()
{

    // empty now
}


// static class members

