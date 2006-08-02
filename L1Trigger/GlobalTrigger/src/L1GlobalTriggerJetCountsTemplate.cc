/**
 * \class L1GlobalTriggerJetCountsTemplate
 * 
 * 
 * 
 * Description: see header file 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder               - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerJetCountsTemplate.h"

// system include files
#include <iostream>
#include <iomanip>
#include <string>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"
//#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"



// forward declarations

// constructor
L1GlobalTriggerJetCountsTemplate::L1GlobalTriggerJetCountsTemplate(
    const std::string &name)
    : L1GlobalTriggerConditions(name) {

    LogDebug ("Trace") 
        << "****Entering " << __PRETTY_FUNCTION__ << " name= " << p_name << std::endl;

}

// copy constructor
void L1GlobalTriggerJetCountsTemplate::copy(const L1GlobalTriggerJetCountsTemplate &cp) {

    p_name = cp.getName();
    setGeEq(cp.getGeEq());

    memcpy(&p_conditionparameter, cp.getConditionParameter(), sizeof(ConditionParameter));
}


L1GlobalTriggerJetCountsTemplate::L1GlobalTriggerJetCountsTemplate(
    const L1GlobalTriggerJetCountsTemplate& cp) 
    : L1GlobalTriggerConditions(cp.p_name) {

    copy(cp);

}

// destructor
L1GlobalTriggerJetCountsTemplate::~L1GlobalTriggerJetCountsTemplate() {

}

// equal operator
L1GlobalTriggerJetCountsTemplate& L1GlobalTriggerJetCountsTemplate::operator= (
    const L1GlobalTriggerJetCountsTemplate& cp) {

    copy(cp);
    return *this;
}

/**
 * setConditionParameter - set the parameters of the condition
 *
 * @param conditionp Pointer to condition parameters.
 *
 */

void L1GlobalTriggerJetCountsTemplate::setConditionParameter(
    const ConditionParameter* conditionp) {
    
    memcpy(&p_conditionparameter, conditionp, sizeof(ConditionParameter));

}


/**
 * blockCondition check if the condition matches.
 *
 * @return The result of the check.
 */

const bool L1GlobalTriggerJetCountsTemplate::blockCondition() const {

//    L1GctJetCounts* jetNr = PSB()->getCountList();
    L1GctJetCounts* jetNr = 0; // TODO un-comment

    const unsigned int nJetCounts = L1GlobalTriggerReadoutRecord::NumberL1JetCounts;
     
    unsigned int jetCount[nJetCounts];
    for (unsigned int i = 0; i < nJetCounts; ++i) {
        jetCount[i] = jetNr->count(i);      		
	}    

// TODO ????

    if (p_conditionparameter.type < nJetCounts) {
        if (!checkThreshold(p_conditionparameter.et_threshold, jetCount[p_conditionparameter.type])) { 
            return false;
        }
    }

    return true;
}
  
void L1GlobalTriggerJetCountsTemplate::printThresholds() const {

    std::cout << "L1GlobalTriggerJetCountsTemplate: Threshold values " << std::endl;
    std::cout << "Condition Name: " << getName() << std::endl;

    std::cout << std::endl;

    std::cout << "Greater equal bit:    " << p_ge_eq << std::endl;

    std::cout << "et_threshold          " << 
        std::hex << p_conditionparameter.et_threshold << std::endl; 
    std::cout << "type		         " 
        << std::dec << p_conditionparameter.type << std::endl; 

    //reset to decimal output
    std::cout << std::dec << std::endl;
        
}
