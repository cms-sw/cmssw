#ifndef GlobalTrigger_L1GlobalTriggerGTL_h
#define GlobalTrigger_L1GlobalTriggerGTL_h

/**
 * \class L1GlobalTriggerGTL
 * 
 * 
 * Description: Global Trigger Logic board.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M. Fierro            - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <bitset>
#include <vector>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations
class L1GlobalTrigger;
class L1GlobalTriggerPSB;

// class declaration
class L1GlobalTriggerGTL
{

public:

    // constructors
    //L1GlobalTriggerGTL();

    L1GlobalTriggerGTL(const L1GlobalTrigger&);

    // destructor
    virtual ~L1GlobalTriggerGTL();

public:

    typedef unsigned int MuonDataWord;

    typedef std::vector<L1MuGMTCand*> L1GmtCandVector;

public:

    /// receive data from Global Muon Trigger
    void receiveGmtObjectData(
        edm::Event&,
        const edm::InputTag&, const int iBxInEvent,
        const bool receiveMu, const unsigned int nrL1Mu);

    /// run the GTL
    void run(edm::Event&, const edm::EventSetup&, const L1GlobalTriggerPSB*, 
        const bool, const int, std::auto_ptr<L1GlobalTriggerObjectMapRecord>&, const unsigned int);

    /// clear GTL
    void reset();

    /// print received Muon dataWord
    void printGmtData(int iBxInEvent) const;

    /// return decision
    inline const std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers>& getDecisionWord() const
    {
        return m_gtlDecisionWord;
    }

    /// return algorithm OR decision
    inline const std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers>& getAlgorithmOR() const
    {
        return m_gtlAlgorithmOR;
    }

    /// return global muon trigger candidate
    inline const L1GmtCandVector* getCandL1Mu() const
    {
        return m_candL1Mu;
    }

private:

    const L1GlobalTrigger& m_GT;

    L1GmtCandVector* m_candL1Mu;

    std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers> m_gtlAlgorithmOR;
    std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers> m_gtlDecisionWord;


};

#endif
