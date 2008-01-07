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

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations
class L1GlobalTrigger;

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
    L1GlobalTriggerPSB(L1GlobalTrigger& gt);

    // destructor
    virtual ~L1GlobalTriggerPSB();

public:

    typedef std::vector<L1GctCand*> L1GctCandVector;

public:

    /// receive Global Calorimeter Trigger objects
    void receiveGctObjectData(
        edm::Event& iEvent,
        const edm::InputTag& caloGctInputTag, const int iBxInEvent,
        const bool receiveNoIsoEG, const unsigned int nrL1NoIsoEG,
        const bool receiveIsoEG, const unsigned int nrL1IsoEG,
        const bool receiveCenJet, const unsigned int nrL1CenJet,
        const bool receiveForJet, const unsigned int nrL1ForJet,
        const bool receiveTauJet, const unsigned int nrL1TauJet,
        const bool receiveETM, const bool receiveETT, const bool receiveHTT,
        const bool receiveJetCounts);

    /// fill the content of active PSB boards
    void fillPsbBlock(
        edm::Event& iEvent,
        const edm::EventSetup& evSetup,
        const int iBxInEvent,
        std::auto_ptr<L1GlobalTriggerReadoutRecord>& gtDaqReadoutRecord);

    /// clear PSB
    void reset();

    /// print Global Calorimeter Trigger data
    void printGctObjectData() const;

    /// pointer to NoIsoEG data list
    inline const L1GctCandVector* getCandL1NoIsoEG() const
    {
        return m_candL1NoIsoEG;
    }

    /// pointer to IsoEG data list
    inline const L1GctCandVector* getCandL1IsoEG() const
    {
        return m_candL1IsoEG;
    }

    /// pointer to CenJet data list
    inline const L1GctCandVector* getCandL1CenJet() const
    {
        return m_candL1CenJet;
    }

    /// pointer to ForJet data list
    inline const L1GctCandVector* getCandL1ForJet() const
    {
        return m_candL1ForJet;
    }

    /// pointer to TauJet data list
    inline const L1GctCandVector* getCandL1TauJet() const
    {
        return m_candL1TauJet;
    }

    /// pointer to ETM data list
    inline L1GctEtMiss* getCandL1ETM() const
    {
        return m_candETM;
    }

    /// pointer to ETT data list
    inline L1GctEtTotal* getCandL1ETT() const
    {
        return m_candETT;
    }

    /// pointer to HTT data list
    inline L1GctEtHad* getCandL1HTT() const
    {
        return m_candHTT;
    }

    /// pointer to JetCounts data list
    inline L1GctJetCounts* getCandL1JetCounts() const
    {
        return m_candJetCounts;
    }

private:

    const L1GlobalTrigger& m_GT;

    L1GctCandVector* m_candL1NoIsoEG;
    L1GctCandVector* m_candL1IsoEG;
    L1GctCandVector* m_candL1CenJet;
    L1GctCandVector* m_candL1ForJet;
    L1GctCandVector* m_candL1TauJet;

    L1GctEtMiss*  m_candETM;
    L1GctEtTotal* m_candETT;
    L1GctEtHad*   m_candHTT;

    L1GctJetCounts* m_candJetCounts;


};

#endif
