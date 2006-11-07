#ifndef GlobalTrigger_L1GlobalTriggerPSB_h
#define GlobalTrigger_L1GlobalTriggerPSB_h
/**
 * \class L1GlobalTriggerPSB
 * 
 * 
 * 
 * Description: Pipelined Synchronising Buffer 
 * Implementation:
 *    <TODO: enter implementation details>
 *    GT PSB receives data from
 *      - Global Calorimeter Trigger
 *      - Technical Trigger
 *   
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
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "FWCore/Framework/interface/Event.h"

// forward declarations
class L1GlobalTrigger;

class L1GctCand;

class L1GctEmCand;
class L1GctJetCand;

class L1GctEtMiss;
class L1GctEtTotal;
class L1GctEtHad;

class L1GctJetCounts;

// class declaration
class L1GlobalTriggerPSB 
{
  
public:
    
    // constructor
    L1GlobalTriggerPSB(L1GlobalTrigger& gt);
    
    // destructor
    virtual ~L1GlobalTriggerPSB();
    
public:
  
    typedef L1GlobalTriggerReadoutSetup::CaloDataWord CaloDataWord;

    typedef std::vector<L1GctCand*> CaloVector;

public:
  
    /// receive input data
    void receiveData(edm::Event&);
    
    /// clear PSB
    void reset();
    
    /// print Global Calorimter Trigger data  
    void printGctData() const;
    
    /// pointer to electron data list 
    inline const CaloVector* getElectronList() const { return glt_electronList; }

    /// pointer to isolated electron data list 
    inline const CaloVector* getIsolatedElectronList() const { return glt_isolatedElectronList; }

    /// pointer to central jet data list 
    inline const CaloVector* getCentralJetList() const { return glt_centralJetList; }

    /// pointer to forward jet data list 
    inline const CaloVector* getForwardJetList() const { return glt_forwardJetList; }

    /// pointer to tau jet data list 
    inline const CaloVector* getTauJetList() const { return glt_tauJetList; }

    /// pointer to Missing Et data list
    inline L1GctEtMiss* getCaloMissingEtList() const { return glt_missingEtList; }

    /// pointer to Total Et data list 
    inline L1GctEtTotal* getCaloTotalEtList() const { return glt_totalEtList; }

    /// pointer to Total Ht data list 
    inline L1GctEtHad* getCaloTotalHtList() const { return glt_totalHtList; }
    
    /// pointer to Jet-counts data list 
    inline L1GctJetCounts* getJetCountsList() const { return glt_jetCountsList; }
    
private:
    
    const L1GlobalTrigger& m_GT;

    CaloVector* glt_electronList;
    CaloVector* glt_isolatedElectronList;
    CaloVector* glt_centralJetList;
    CaloVector* glt_forwardJetList;
    CaloVector* glt_tauJetList;

    L1GctEtMiss*  glt_missingEtList;
    L1GctEtTotal* glt_totalEtList;
    L1GctEtHad*   glt_totalHtList;
    
    L1GctJetCounts* glt_jetCountsList;


};
    
#endif
