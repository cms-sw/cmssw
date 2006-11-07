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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

// system include files
#include <iostream>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"

// forward declarations


// constructor
L1GlobalTriggerFDL::L1GlobalTriggerFDL(
    L1GlobalTrigger& gt) 
    : m_GT(gt) {

    // create empty FDL word        
    m_gtFdlWord = new L1GtFdlWord();

}

// destructor
L1GlobalTriggerFDL::~L1GlobalTriggerFDL() { 

    reset();    
    delete m_gtFdlWord; 
  
}

// Operations

// run FDL
void L1GlobalTriggerFDL::run(int iBxInEvent) {

    const L1GlobalTriggerConfig* gtConf = m_GT.gtSetup()->gtConfig();


    const unsigned int numberTriggerBits = 
        L1GlobalTriggerReadoutSetup::NumberPhysTriggers;

    // get gtlDecisionWord from GTL
    std::bitset<numberTriggerBits> 
        gtlDecisionWord = m_GT.gtGTL()->getAlgorithmOR();

    // add trigger mask            
    std::bitset<numberTriggerBits> fdlDecisionWord;
      
    if (gtConf != 0) {
        fdlDecisionWord = gtlDecisionWord & ~(gtConf->getTriggerMask());
    }
    
        // [ ... convert decision word from std::bitset to std::vector<bool>
        //       TODO remove this block when changing DecisionWord to std::bitset    

    
    L1GlobalTriggerReadoutSetup::DecisionWord fdlDecisionWordVec(numberTriggerBits);    

    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {
        
        bool bitValue = fdlDecisionWord.test( iBit );        
        fdlDecisionWordVec[ iBit ] = bitValue;   
    }
    
        // ... ]
    
    // fill everything we know in the L1GtFdlWord
    
    // BxInEvent
    m_gtFdlWord->setBxInEvent(iBxInEvent);
    
    // decision word
    m_gtFdlWord->setGtDecisionWord(fdlDecisionWordVec);

    // finalOR
    // TODO FIXME set DAQ partition where L1A is sent; now: hardwired, first partition
    int daqPartitionL1A = 0;  
    uint16_t finalOrValue = 0;
    
    if ( fdlDecisionWord.any() ) {
        finalOrValue = 1 << daqPartitionL1A; 
    } 

    m_gtFdlWord->setFinalOR(finalOrValue);        
    
    //
    
    

}

// clear FDL
void L1GlobalTriggerFDL::reset() {
    
    m_gtFdlWord->reset();
    
}
