/**
 * \class L1GtConditionEvaluation
 * 
 * 
 * Description: Base class for evaluation of the L1 Global Trigger object templates.
 * 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna 
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

// system include files

// user include files

//   base class

//
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtConditionEvaluation::L1GtConditionEvaluation()
{
    m_condMaxNumberObjects = 0;
    m_condLastResult = false;
    m_combinationsInCond = new CombinationsInCond;

}

// copy constructor
L1GtConditionEvaluation::L1GtConditionEvaluation(L1GtConditionEvaluation& cp)
{

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

}

// destructor
L1GtConditionEvaluation::~L1GtConditionEvaluation()
{

    delete m_combinationsInCond;

}

// methods

