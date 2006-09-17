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

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

// system include files
#include <iostream>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"

// forward declarations


// constructor
L1GlobalTriggerFDL::L1GlobalTriggerFDL(
    L1GlobalTrigger& gt) 
    : m_GT(gt) {

}

// destructor
L1GlobalTriggerFDL::~L1GlobalTriggerFDL() { 

    reset();
  
}

// Operations

// run FDL
void L1GlobalTriggerFDL::run() {

    const L1GlobalTriggerConfig* gtConf = m_GT.gtSetup()->gtConfig();

    std::bitset<L1GlobalTriggerReadoutRecord::NumberPhysTriggers> 
        gtlDecision = m_GT.gtGTL()->get_algorithmOR();
    
    if (gtConf != 0) {
        theDecisionWord =  gtlDecision & ~(gtConf->getTriggerMask()); 
    } else {
        theDecisionWord = gtlDecision;
    }
  
    theDecision = theDecisionWord.any();

}

// clear FDL
void L1GlobalTriggerFDL::reset() {
    
    theDecisionWord.reset();
    theDecision = false;

}
