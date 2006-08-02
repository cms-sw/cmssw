/**
 * \class L1GlobalTriggerEsumsTemplate
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
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerEsumsTemplate.h"

// system include files
#include <iostream>
#include <iomanip>
#include <string>

// user include files
//   base class
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
//#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations


// constructor
L1GlobalTriggerEsumsTemplate::L1GlobalTriggerEsumsTemplate( 
    const std::string &name)
    : L1GlobalTriggerConditions(name) {

    LogDebug ("Trace") 
        << "****Entering " << __PRETTY_FUNCTION__ << " name= " << p_name 
        << std::endl;

}

// copy constructor
void L1GlobalTriggerEsumsTemplate::copy(const L1GlobalTriggerEsumsTemplate &cp) {

    p_name = cp.getName();
    p_sumtype = cp.p_sumtype;
    setGeEq(cp.getGeEq());

    memcpy(&p_conditionparameter, cp.getConditionParameter(), sizeof(ConditionParameter));
}


L1GlobalTriggerEsumsTemplate::L1GlobalTriggerEsumsTemplate(
    const L1GlobalTriggerEsumsTemplate& cp) 
    : L1GlobalTriggerConditions(cp.p_name) {

    copy(cp);

}

// destructor
L1GlobalTriggerEsumsTemplate::~L1GlobalTriggerEsumsTemplate() {

}


// equal operator
L1GlobalTriggerEsumsTemplate& L1GlobalTriggerEsumsTemplate::operator= (
    const L1GlobalTriggerEsumsTemplate& cp) {

    copy(cp);
    return *this;
}

/**
 * setConditionParameter - set the parameters of the condition
 *
 * @param conditionp Pointer to condition parameters.
 * @param st Type of the sum in this condition. (etm ett or htt)
 */

void L1GlobalTriggerEsumsTemplate::setConditionParameter(
    const ConditionParameter* conditionp, SumType st) {

    p_sumtype = st;    
    memcpy(&p_conditionparameter, conditionp, sizeof(ConditionParameter));
}


/**
 * blockCondition - Check if the condition matches.
 *
 * @return Boolean result of the check.
 *
 */
 
const bool L1GlobalTriggerEsumsTemplate::blockCondition() const {

    unsigned int candEt = 0;    
    unsigned int candPhi = 0;

    // get energy and phi (ETM only) for the trigger object 
    // TODO put back the candidates when PSB available

    switch (p_sumtype) {
		case ETT:
        {
//          L1GctEtTotal* cand1 = PSB->getCaloTotalEtList();
            L1GctEtTotal* cand1 = 0;        
            if (cand1 == 0) return false;
            
            candEt = cand1->et();			
			break;
        }
        case ETM:
        {
//        L1GctEtMiss* cand2 = PSB()->getCaloMissingEtList();
            L1GctEtMiss* cand2 = 0;        
            if (cand2 == 0) return false;
            
            candEt  = cand2->et();
            candPhi = cand2->phi();            
            break;
        }
        case HTT:
        {
//        L1GctEtHad* cand3 = PSB()->getCaloHadEtList();
            L1GctEtHad* cand3 = 0;        
            if (cand3 == 0) return false;
            
            candEt = cand3->et();            
            break;
        }
		default:
            // should not arrive here
            return false;
			break;
	}

    // check et threshold
    if (!checkThreshold(p_conditionparameter.et_threshold, candEt)) return false;

    /// for etm check phi also
    if (p_sumtype == ETM) {
        if (!checkBit(p_conditionparameter.phi, candPhi)) return false;
    }

    // condition matches    
    return true;
}
    

void L1GlobalTriggerEsumsTemplate::printThresholds() const {

    std::cout << "L1GlobalTriggerEsumsTemplate: Threshold values " << std::endl;
    std::cout << "Condition Name: " << getName() << std::endl;

    std::cout << "Type of Sum: ";
    switch (p_sumtype) {
		case ETM:
            std::cout << "etm";			
			break;
        case ETT:
            std::cout << "ett";         
            break;
        case HTT:
            std::cout << "htt";         
            break;
		default:
            // nothing
			break;
	}
    std::cout << std::endl;

    std::cout << "Greater equal bit: " << p_ge_eq << std::endl;

    std::cout << "et_threshold          " 
        << std::hex << p_conditionparameter.et_threshold << std::endl; 
    std::cout << "en_overflow           " 
        << std::hex << p_conditionparameter.en_overflow << std::endl; 

    if (p_sumtype == ETM) {
        std::cout << "phi               " 
            << std::hex << p_conditionparameter.phi << std::endl;
    }
      
     // reset to decimal output
     std::cout << std::dec << std::endl;      
}


