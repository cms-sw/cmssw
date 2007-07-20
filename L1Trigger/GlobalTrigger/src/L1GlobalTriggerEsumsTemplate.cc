/**
 * \class L1GlobalTriggerEsumsTemplate
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
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerEsumsTemplate.h"

// system include files
#include <iostream>
#include <iomanip>
#include <string>

// user include files
//   base class
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations


// constructor
L1GlobalTriggerEsumsTemplate::L1GlobalTriggerEsumsTemplate(
    const L1GlobalTrigger& gt,
    const std::string& name)
        : L1GlobalTriggerConditions(gt, name)
{

    //    LogDebug ("Trace")
    //        << "****Entering " << __PRETTY_FUNCTION__ << " name= " << p_name
    //        << std::endl;

}

// copy constructor
void L1GlobalTriggerEsumsTemplate::copy(const L1GlobalTriggerEsumsTemplate &cp)
{

    p_name = cp.getName();
    p_sumtype = cp.p_sumtype;
    setGeEq(cp.getGeEq());

    memcpy(&p_conditionparameter, cp.getConditionParameter(), sizeof(ConditionParameter));
}


L1GlobalTriggerEsumsTemplate::L1GlobalTriggerEsumsTemplate(
    const L1GlobalTriggerEsumsTemplate& cp)
        : L1GlobalTriggerConditions(cp.m_GT, cp.p_name)
{

    copy(cp);

}

// destructor
L1GlobalTriggerEsumsTemplate::~L1GlobalTriggerEsumsTemplate()
{

    // empty

}


// equal operator
L1GlobalTriggerEsumsTemplate& L1GlobalTriggerEsumsTemplate::operator= (
    const L1GlobalTriggerEsumsTemplate& cp)
{

    //    m_GT = cp.m_GT; // TODO uncomment ???
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
    const ConditionParameter* conditionp, SumType st)
{

    p_sumtype = st;
    memcpy(&p_conditionparameter, conditionp, sizeof(ConditionParameter));
}


/**
 * blockCondition - Check if the condition matches.
 *
 * @return Boolean result of the check.
 *
 */

const bool L1GlobalTriggerEsumsTemplate::blockCondition() const
{

    // store for completness the indices of the calorimeter objects
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;

    // clear the p_combinationsInCond vector
    (*p_combinationsInCond).clear();

    // clear the p_objectsInCond vector
    (*p_objectsInCond).clear();

    ObjectTypeInCond typeInCond;


    unsigned int candEt = 0;
    unsigned int candPhi = 0;

    // get energy and phi (ETM only) for the trigger object

    switch (p_sumtype) {
        case ETT_ST: {
                L1GctEtTotal* cand1 = m_GT.gtPSB()->getCaloTotalEtList();

                typeInCond.push_back(ETT);
                (*p_objectsInCond) = typeInCond;

                if (cand1 == 0) {
                    return false;
                }

                candEt = cand1->et();

                break;
            }
        case ETM_ST: {
                L1GctEtMiss* cand2 = m_GT.gtPSB()->getCaloMissingEtList();

                typeInCond.push_back(ETM);
                (*p_objectsInCond) = typeInCond;

                if (cand2 == 0) {
                    return false;
                }

                candEt  = cand2->et();
                candPhi = cand2->phi();

                break;
            }
        case HTT_ST: {
                L1GctEtHad* cand3 = m_GT.gtPSB()->getCaloTotalHtList();

                typeInCond.push_back(HTT);
                (*p_objectsInCond) = typeInCond;

                if (cand3 == 0) {
                    return false;
                }

                candEt = cand3->et();


                break;
            }
        default:
            // should not arrive here
            return false;
            break;
    }

    // check et threshold
    if (!checkThreshold(p_conditionparameter.et_threshold, candEt)) {
        return false;
    }

    /// for etm check phi also
    if (p_sumtype == ETM_ST) {

        // phi bitmask is saved in two u_int64_t (see parser)
        if (candPhi < 64) {
            if (!checkBit(p_conditionparameter.phi0word, candPhi) ) {
                return false;
            }
        } else {
            if (!checkBit(p_conditionparameter.phi1word, candPhi - 64)) {
                return false;
            }
        }


    }

    // condition matches

    // index is always zero, as they are global quantities (there is only one object)
    int indexEsum = 0;

    objectsInComb.push_back(indexEsum);
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

    LogTrace("L1GlobalTriggerEsumsTemplate")
    << "\n  List of combinations passing all requirements for this condition: \n  "
    <<  myCout1.str()
    << " \n"
    << std::endl;


    return true;
}


void L1GlobalTriggerEsumsTemplate::printThresholds(std::ostream& myCout) const
{

    myCout << "L1GlobalTriggerEsumsTemplate: threshold values " << std::endl;
    myCout << "Condition Name: " << getName() << std::endl;

    switch (p_sumtype) {
        case ETM_ST:
            myCout << "Type of Sum: " << "ETM";
            break;
        case ETT_ST:
            myCout << "Type of Sum: " << "ETT";
            break;
        case HTT_ST:
            myCout << "Type of Sum: " << "HTT";
            break;
        default:
            // nothing
            break;
    }

    myCout << "\ngreater or equal bit: " << p_ge_eq << std::endl;

    myCout << "\n  TEMPLATE " << "0" // only one
    << std::endl;
    myCout << "    et_threshold          "
    << std::hex << p_conditionparameter.et_threshold
    << std::endl;
    myCout << "    en_overflow           "
    << std::hex << p_conditionparameter.en_overflow << std::endl;
    if (p_sumtype == ETM_ST) {
        myCout << "    phi0word                   "
        << std::hex << p_conditionparameter.phi0word << std::endl;
        myCout << "    phi1word                   "
        << std::hex << p_conditionparameter.phi1word << std::endl;
    }

    // reset to decimal output
    myCout << std::dec << std::endl;
}


