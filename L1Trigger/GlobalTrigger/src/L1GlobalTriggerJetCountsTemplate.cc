/**
 * \class L1GlobalTriggerJetCountsTemplate
 * 
 * 
 * Description: see header file.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder               - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerJetCountsTemplate.h"

// system include files
#include <iostream>
#include <iomanip>
#include <string>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations

// constructor
L1GlobalTriggerJetCountsTemplate::L1GlobalTriggerJetCountsTemplate(
    const L1GlobalTrigger& gt,
    const std::string& name)
    : L1GlobalTriggerConditions(gt, name) {

//    LogDebug ("Trace") 
//        << "****Entering " << __PRETTY_FUNCTION__ << " name= " << p_name << std::endl;

}

// copy constructor
void L1GlobalTriggerJetCountsTemplate::copy(const L1GlobalTriggerJetCountsTemplate &cp) {

    p_name = cp.getName();
    setGeEq(cp.getGeEq());

    memcpy(&p_conditionparameter, cp.getConditionParameter(), sizeof(ConditionParameter));
}


L1GlobalTriggerJetCountsTemplate::L1GlobalTriggerJetCountsTemplate(
    const L1GlobalTriggerJetCountsTemplate& cp) 
    : L1GlobalTriggerConditions(cp.m_GT, cp.p_name) {

    copy(cp);

}

// destructor
L1GlobalTriggerJetCountsTemplate::~L1GlobalTriggerJetCountsTemplate() {

}

// equal operator
L1GlobalTriggerJetCountsTemplate& L1GlobalTriggerJetCountsTemplate::operator= (
    const L1GlobalTriggerJetCountsTemplate& cp) {

//    m_GT = cp.m_GT; // TODO uncomment ???
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

    // store for completness the indices of the calorimeter objects
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;

    // clear the p_combinationsInCond vector
    (*p_combinationsInCond).clear();

    // clear the p_objectsInCond vector
    (*p_objectsInCond).clear();

    ObjectTypeInCond typeInCond;

    // get jet counts and set the type
    L1GctJetCounts* jetNr = m_GT.gtPSB()->getListJetCounts();

    typeInCond.push_back(JetCounts);
    (*p_objectsInCond) = typeInCond;

    const unsigned int nJetCounts = L1GlobalTriggerReadoutSetup::NumberL1JetCounts;
     
    unsigned int jetCount[nJetCounts];
    for (unsigned int i = 0; i < nJetCounts; ++i) {
        jetCount[i] = jetNr->count(i);      		
	}    

    if (p_conditionparameter.type < nJetCounts) {
        if (!checkThreshold(p_conditionparameter.et_threshold, jetCount[p_conditionparameter.type])) { 
            return false;
        }
    }

    // condition matches

    // index is always zero, as jet counts is a global quantity (there is only one object)
    int indexJetCounts = 0;

    objectsInComb.push_back(indexJetCounts);
    (*p_combinationsInCond).push_back(objectsInComb);

    CombinationsInCond::const_iterator itVV;
    std::ostringstream myCout1;

    for(itVV  = (*p_combinationsInCond).begin();
            itVV != (*p_combinationsInCond).end(); itVV++) {

        myCout1 << "( ";

        std::copy((*itVV).begin(), (*itVV).end(),
                  std::ostream_iterator<int> (myCout1, " "));

        myCout1 << "); ";

    }

    LogTrace("L1GlobalTriggerJetCountsTemplate")
    << "\n  List of combinations passing all requirements for this condition: \n  "
    <<  myCout1.str()
    << " \n"
    << std::endl;

    return true;
}
  
void L1GlobalTriggerJetCountsTemplate::printThresholds(std::ostream& myCout) const {

    myCout << "L1GlobalTriggerJetCountsTemplate: threshold values " << std::endl;
    myCout << "Condition Name: " << getName() << std::endl;

    myCout << std::endl;

    myCout << "Greater equal bit:    " 
        << p_ge_eq << std::endl;

    myCout << "et_threshold          " << 
        std::hex << p_conditionparameter.et_threshold << std::endl; 
    myCout << "type		         " 
        << std::dec << p_conditionparameter.type << std::endl; 

    //reset to decimal output
    myCout << std::dec << std::endl;
        
}
