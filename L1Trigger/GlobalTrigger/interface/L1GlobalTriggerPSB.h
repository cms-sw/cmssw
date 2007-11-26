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

    typedef std::vector<L1GctCand*> CaloVector;

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
    inline const CaloVector* getListNoIsoEG() const
    {
        return m_listNoIsoEG;
    }

    /// pointer to IsoEG data list
    inline const CaloVector* getListIsoEG() const
    {
        return m_listIsoEG;
    }

    /// pointer to CenJet data list
    inline const CaloVector* getListCenJet() const
    {
        return m_listCenJet;
    }

    /// pointer to ForJet data list
    inline const CaloVector* getListForJet() const
    {
        return m_listForJet;
    }

    /// pointer to TauJet data list
    inline const CaloVector* getListTauJet() const
    {
        return m_listTauJet;
    }

    /// pointer to ETM data list
    inline L1GctEtMiss* getListETM() const
    {
        return m_listETM;
    }

    /// pointer to ETT data list
    inline L1GctEtTotal* getListETT() const
    {
        return m_listETT;
    }

    /// pointer to HTT data list
    inline L1GctEtHad* getListHTT() const
    {
        return m_listHTT;
    }

    /// pointer to JetCounts data list
    inline L1GctJetCounts* getListJetCounts() const
    {
        return m_listJetCounts;
    }

private:

    const L1GlobalTrigger& m_GT;

    CaloVector* m_listNoIsoEG;
    CaloVector* m_listIsoEG;
    CaloVector* m_listCenJet;
    CaloVector* m_listForJet;
    CaloVector* m_listTauJet;

    L1GctEtMiss*  m_listETM;
    L1GctEtTotal* m_listETT;
    L1GctEtHad*   m_listHTT;

    L1GctJetCounts* m_listJetCounts;


};

#endif
