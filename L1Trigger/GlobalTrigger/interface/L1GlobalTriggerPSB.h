#ifndef GlobalTrigger_L1GlobalTriggerPSB_h
#define GlobalTrigger_L1GlobalTriggerPSB_h

/**
 * \class L1GlobalTriggerPSB
 * 
 * 
 * Description: Pipelined Synchronising Buffer.  
 *
 * Implementation:
 *    GT PSB receives data from
 *      - Global Calorimeter Trigger
 *      - Technical Trigger
 *      
 * \author: M. Fierro            - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <vector>
#include <boost/cstdint.hpp>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTriggerRecord.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/Selector.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"


// forward declarations
class L1GctCand;

class L1GctEmCand;
class L1GctJetCand;

class L1GctEtMiss;
class L1GctEtTotal;
class L1GctEtHad;

class L1GctJetCounts;

class L1GlobalTriggerReadoutRecord;

// class declaration
class L1GlobalTriggerPSB
{

public:

    // constructor
    L1GlobalTriggerPSB();

    // string for the selector label edm::ModuleLabelSelector
    L1GlobalTriggerPSB(const std::string selLabel);

    // destructor
    virtual ~L1GlobalTriggerPSB();

public:

    /// initialize the class (mainly reserve)
    void init(const int nrL1NoIsoEG, const int nrL1IsoEG, 
            const int nrL1CenJet, const int nrL1ForJet, const int nrL1TauJet,
            const int numberTechnicalTriggers);

    /// receive Global Calorimeter Trigger objects
    void receiveGctObjectData(
        edm::Event& iEvent,
        const edm::InputTag& caloGctInputTag, const int iBxInEvent,
        const bool receiveNoIsoEG, const int nrL1NoIsoEG,
        const bool receiveIsoEG, const int nrL1IsoEG,
        const bool receiveCenJet, const int nrL1CenJet,
        const bool receiveForJet, const int nrL1ForJet,
        const bool receiveTauJet, const int nrL1TauJet,
        const bool receiveETM, const bool receiveETT, const bool receiveHTT,
        const bool receiveJetCounts);

    /// receive technical trigger
    void receiveTechnicalTriggers(edm::Event& iEvent,
        const edm::InputTag& technicalTriggersInputTag,
        const int iBxInEvent, const bool receiveTechTr,
        const int nrL1TechTr);

    /// fill the content of active PSB boards
    void fillPsbBlock(
        edm::Event& iEvent,
        const boost::uint16_t& activeBoardsGtDaq,
        const std::vector<L1GtBoard>& boardMaps,
        const int iBxInEvent,
        std::auto_ptr<L1GlobalTriggerReadoutRecord>& gtDaqReadoutRecord);

    /// clear PSB
    void reset();

    /// print Global Calorimeter Trigger data
    void printGctObjectData(const int iBxInEvent) const;

    /// pointer to NoIsoEG data list
    inline const std::vector<const L1GctCand*>* getCandL1NoIsoEG() const
    {
        return m_candL1NoIsoEG;
    }

    /// pointer to IsoEG data list
    inline const std::vector<const L1GctCand*>* getCandL1IsoEG() const
    {
        return m_candL1IsoEG;
    }

    /// pointer to CenJet data list
    inline const std::vector<const L1GctCand*>* getCandL1CenJet() const
    {
        return m_candL1CenJet;
    }

    /// pointer to ForJet data list
    inline const std::vector<const L1GctCand*>* getCandL1ForJet() const
    {
        return m_candL1ForJet;
    }

    /// pointer to TauJet data list
    inline const std::vector<const L1GctCand*>* getCandL1TauJet() const
    {
        return m_candL1TauJet;
    }

    /// pointer to ETM data list
    inline const L1GctEtMiss* getCandL1ETM() const
    {
        return m_candETM;
    }

    /// pointer to ETT data list
    inline const L1GctEtTotal* getCandL1ETT() const
    {
        return m_candETT;
    }

    /// pointer to HTT data list
    inline const L1GctEtHad* getCandL1HTT() const
    {
        return m_candHTT;
    }

    /// pointer to JetCounts data list
    inline const L1GctJetCounts* getCandL1JetCounts() const
    {
        return m_candJetCounts;
    }

    /// pointer to technical trigger bits 
    inline const std::vector<bool>* getGtTechnicalTriggers() const
    {
        return &m_gtTechnicalTriggers;
    }

private:

    std::vector<const L1GctCand*>* m_candL1NoIsoEG;
    std::vector<const L1GctCand*>* m_candL1IsoEG;
    std::vector<const L1GctCand*>* m_candL1CenJet;
    std::vector<const L1GctCand*>* m_candL1ForJet;
    std::vector<const L1GctCand*>* m_candL1TauJet;

    const L1GctEtMiss*  m_candETM;
    const L1GctEtTotal* m_candETT;
    const L1GctEtHad*   m_candHTT;

    const L1GctJetCounts* m_candJetCounts;
    
    /// technical trigger bits 
    std::vector<bool> m_gtTechnicalTriggers;
    
    /// handles to the technical trigger records
    std::vector<edm::Handle<L1GtTechnicalTriggerRecord> > m_techTrigRecords; 
    
    /// selector for getMany methods
    edm::Selector m_techTrigSelector;    

};

#endif
