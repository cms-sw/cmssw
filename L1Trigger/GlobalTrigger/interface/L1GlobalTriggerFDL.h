#ifndef GlobalTrigger_L1GlobalTriggerFDL_h
#define GlobalTrigger_L1GlobalTriggerFDL_h
/**
 * \class L1GlobalTriggerFDL
 * 
 * 
 * 
 * Description: Final Decision Logic board 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M. Fierro            - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include <vector>
#include <bitset>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// forward declarations
class L1GlobalTrigger;

// class declaration
class L1GlobalTriggerFDL {
  
public: 

    /// constructor
    L1GlobalTriggerFDL(L1GlobalTrigger& gt);
  
    /// destructor
    virtual ~L1GlobalTriggerFDL();
  
    /// run the FDL
    void run();
  
    /// clear FDL
    void reset(); 

    /// return decision word
    inline const std::bitset<L1GlobalTriggerReadoutRecord::NumberPhysTriggers>& getDecisionWord() const { return theDecisionWord; }

    /// return global decision
    inline bool getDecision() const { return theDecision; }
 
  private:

    const L1GlobalTrigger& m_GT;
    
    std::bitset<L1GlobalTriggerReadoutRecord::NumberPhysTriggers> theDecisionWord;
    bool theDecision;

};
  
#endif
