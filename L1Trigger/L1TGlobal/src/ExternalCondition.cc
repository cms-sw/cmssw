/**
 * \class ExternalCondition
 *
 *
 * Description: evaluation of a CondExternal condition.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/ExternalCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/ExternalTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
l1t::ExternalCondition::ExternalCondition() :
    ConditionEvaluation() {

    //empty

}

//     from base template condition (from event setup usually)
l1t::ExternalCondition::ExternalCondition(const GlobalCondition* eSumTemplate, const GlobalBoard* ptrGTB) :
    ConditionEvaluation(),
    m_gtExternalTemplate(static_cast<const ExternalTemplate*>(eSumTemplate)),
    m_uGtB(ptrGTB)

{

    // maximum number of objects received for the evaluation of the condition
    // energy sums are global quantities - one object per event

    m_condMaxNumberObjects = 1;  //blw ???

}

// copy constructor
void l1t::ExternalCondition::copy(const l1t::ExternalCondition &cp) {

    m_gtExternalTemplate = cp.gtExternalTemplate();
    m_uGtB = cp.getuGtB();

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

l1t::ExternalCondition::ExternalCondition(const l1t::ExternalCondition& cp) :
    ConditionEvaluation() {

    copy(cp);

}

// destructor
l1t::ExternalCondition::~ExternalCondition() {

    // empty

}

// equal operator
l1t::ExternalCondition& l1t::ExternalCondition::operator= (const l1t::ExternalCondition& cp)
{
    copy(cp);
    return *this;
}

// methods
void l1t::ExternalCondition::setGtExternalTemplate(const ExternalTemplate* eSumTempl) {

    m_gtExternalTemplate = eSumTempl;

}

///   set the pointer to uGT GlobalBoard
void l1t::ExternalCondition::setuGtB(const GlobalBoard* ptrGTB) {

    m_uGtB = ptrGTB;

}

// try all object permutations and check spatial correlations, if required
const bool l1t::ExternalCondition::evaluateCondition(const int bxEval) const {



    LogDebug("L1TGlobal") << "Evaluating External Condition " 
                          <<  m_gtExternalTemplate->condName() 
			  << " on Channel "  << m_gtExternalTemplate->extChannel() 
			  << " relative Bx " << m_gtExternalTemplate->condRelativeBx() << std::endl;
    // number of trigger objects in the condition
    // in fact, there is only one object
//    int iCondition = 0;

    // condition result condResult set to true if the energy sum
    // passes all requirements
    bool condResult = false;

    // store the indices of the calorimeter objects
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;

    // clear the m_combinationsInCond vector
    (combinationsInCond()).clear();

    // clear the indices in the combination
    objectsInComb.clear();

    const BXVector<const GlobalExtBlk*>* candVec = m_uGtB->getCandL1External();

    // Look at objects in bx = bx + relativeBx
    int useBx = bxEval + m_gtExternalTemplate->condRelativeBx();
    unsigned int exCondCh = m_gtExternalTemplate->extChannel(); 

    // Fail condition if attempting to get Bx outside of range
    if( ( useBx < candVec->getFirstBX() ) ||
	( useBx > candVec->getLastBX() ) ) {
      return false;
    }

    int numberObjects = candVec->size(useBx);
    if (numberObjects < 1) {
        return false;
    }

    //get external block (should only be one for the bx)
    GlobalExtBlk ext = *(candVec->at(useBx,0));
    //ext.print(std::cout);

    // check external bit
    if ( !ext.getExternalDecision(exCondCh) ) {
      LogDebug("L1TGlobal") << "\t\t External Condition was not set" << std::endl;
        return false;
    }

    // index is always zero, as they are global quantities (there is only one object)
    int indexObj = 0;

    //Do we need this?
    objectsInComb.push_back(indexObj);
    (combinationsInCond()).push_back(objectsInComb);

    // if we get here all checks were successfull for this combination
    // set the general result for evaluateCondition to "true"
    condResult = true;
    LogDebug("L1TGlobal") << "\t\t Congrats, External Condition was set!" << std::endl;
    
    return condResult;

}

void l1t::ExternalCondition::print(std::ostream& myCout) const {

    m_gtExternalTemplate->print(myCout);
    ConditionEvaluation::print(myCout);

}

