#ifndef GlobalTrigger_L1GlobalTriggerGTL_h
#define GlobalTrigger_L1GlobalTriggerGTL_h
/**
 * \class L1GlobalTriggerGTL
 * 
 * 
 * 
 * Description: Global Trigger Logic board
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
#include <set>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"

#include "FWCore/Framework/interface/Event.h"

// forward declarations
class L1GlobalTrigger;
class L1GlobalTriggerConditions;
class L1GlobalTriggerConfig;

// class declaration
class L1GlobalTriggerGTL 
{

public:

    // constructor
    L1GlobalTriggerGTL(const L1GlobalTrigger&);
    
    // destructor
    virtual ~L1GlobalTriggerGTL();
    
public:
  
    typedef unsigned int MuonDataWord;
    
    typedef std::vector<L1MuGMTCand*> GMTVector;

    // TODO particleBlock: set of unsigned int instead of std::string?
    typedef std::set<unsigned int> particleBlock; 
//    typedef std::set<std::string> particleBlock; 

    typedef std::vector<particleBlock> algoVector;

    typedef std::vector<L1GlobalTriggerConditions*> conditions;

    typedef std::vector<conditions*> conditionContainer;

public:

    /// receive data from Global Muon Trigger
    void receiveData(edm::Event&, int iBxInEvent);
    
    /// run the GTL
    void run(int iBxInEvent);
    
    /// clear GTL
    void reset(); 
    
    /// print received Muon dataWord
    void printGmtData(int iBxInEvent) const;

    /// return decision
    inline const std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers>& getDecisionWord() const { return m_gtlDecisionWord; }
    
    /// return algorithm OR decision
    inline const std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers>& getAlgorithmOR() const { return m_gtlAlgorithmOR; }
    
    /// return muon decision
    inline const std::bitset<L1GlobalTriggerSetup::MaxItem>& getDecision_MUON() const { return glt_cond[0]; }

    /// return electon/gamma decision
    inline const std::bitset<L1GlobalTriggerSetup::MaxItem>& getDecision_EG() const { return glt_cond[1]; }

    /// return isolated electron/gamma decision
    inline const std::bitset<L1GlobalTriggerSetup::MaxItem>& getDecision_IEG() const { return glt_cond[2]; }
    
    /// return central jet decision
    inline const std::bitset<L1GlobalTriggerSetup::MaxItem>& getDecision_CJET() const { return glt_cond[3]; }
    
    /// return forward jet decision
    inline const std::bitset<L1GlobalTriggerSetup::MaxItem>& getDecision_FJET() const { return glt_cond[4]; }

    /// return tau jet decision
    inline const std::bitset<L1GlobalTriggerSetup::MaxItem>& getDecision_TJET() const { return glt_cond[5]; }

    /// return total transverse energy decision
    inline const std::bitset<L1GlobalTriggerSetup::MaxItem>& getDecision_ETT() const { return glt_cond[6]; }

    /// return missing transverse energy decision
    inline const std::bitset<L1GlobalTriggerSetup::MaxItem>& getDecision_ETM() const { return glt_cond[7]; }
    
    /// return hadron transverse energy decision
    inline const std::bitset<L1GlobalTriggerSetup::MaxItem>& getDecision_HTT() const { return glt_cond[8]; }

    /// return jet counts decision
    inline const std::bitset<L1GlobalTriggerSetup::MaxItem>& getDecision_JC() const { return glt_cond[9]; }
    
// TODO un-comment if I decide to keep the MenuItem enum
//    /// return decision
//    inline const bool getDecision(L1GlobalTriggerSetup::MenuItem item) const { return m_gtlDecisionWord.element(item); }
    
    /// return global muon trigger candidate data words
    const std::vector< MuonDataWord > getMuons() const;

    /// return global muon trigger candidate  
    inline const GMTVector* getMuonCandidates() const { return glt_muonCand; }
    
private:

    const L1GlobalTrigger& m_GT;

    GMTVector* glt_muonCand;
    
    std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers> m_gtlAlgorithmOR;
    std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers> m_gtlDecisionWord;

    std::bitset<L1GlobalTriggerSetup::MaxItem> glt_cond[9];
    
    std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers> glt_generalAND;
    
    algoVector glt_algos;

    conditionContainer glt_particleConditions;

};

#endif
