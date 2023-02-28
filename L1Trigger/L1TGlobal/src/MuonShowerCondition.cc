/**                                                                                                             
 * \class MuonShowerCondition                                                                                
 *                                                                                                               
 *                                                                                                                  
 * Description: evaluation of High Multiplicity Triggers (HMTs) based on the presence and type of a muon shower.                             
 *                                                                                                                      
 * Implementation:                                                                                                     
 *    This condition class checks for the presente of a valid muon shower in the event.                                                                  
 *    If present, according to the condition parsed by the xml menu 
 *    (four possibilities for the first Run 3 implementation: MuonShower0, MuonShower1, MuonShowerOutOfTime0, MuonShowerOutOfTime1)
 *    the corresponding boolean flag is checked (isOneNominalInTime, isOneTightInTime, musOutOfTime0, musOutOfTime1).   
 *    If it is set to 1, the condition is satisfied and the object  is saved.
 *    Note that for the start of Run 3 only two cases are considered in the menu: Nominal and Tight muon showers.  
 *  
 * \author: S. Dildick (2021) - Rice University                                                    
 *         
 * \fixes by: E. Fontanesi, E. Yigitbasi, A. Loeliger (2023)                                                                                                
 *         
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/MuonShowerCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/MuonShowerTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"
#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
l1t::MuonShowerCondition::MuonShowerCondition() : ConditionEvaluation() {
  // empty
}

//     from base template condition (from event setup usually)
l1t::MuonShowerCondition::MuonShowerCondition(const GlobalCondition* muonShowerTemplate,
                                              const GlobalBoard* ptrGTL,
                                              const int nrL1MuShower)
    : ConditionEvaluation(),
      m_gtMuonShowerTemplate(static_cast<const MuonShowerTemplate*>(muonShowerTemplate)),
      m_gtGTL(ptrGTL) {
  m_condMaxNumberObjects = nrL1MuShower;
}

// copy constructor
void l1t::MuonShowerCondition::copy(const l1t::MuonShowerCondition& cp) {
  m_gtMuonShowerTemplate = cp.gtMuonShowerTemplate();
  m_gtGTL = cp.gtGTL();

  m_condMaxNumberObjects = cp.condMaxNumberObjects();
  m_condLastResult = cp.condLastResult();
  m_combinationsInCond = cp.getCombinationsInCond();

  m_verbosity = cp.m_verbosity;
}

l1t::MuonShowerCondition::MuonShowerCondition(const l1t::MuonShowerCondition& cp) : ConditionEvaluation() { copy(cp); }

// destructor
l1t::MuonShowerCondition::~MuonShowerCondition() {
  // empty
}

// equal operator
l1t::MuonShowerCondition& l1t::MuonShowerCondition::operator=(const l1t::MuonShowerCondition& cp) {
  copy(cp);
  return *this;
}

// methods
void l1t::MuonShowerCondition::setGtMuonShowerTemplate(const MuonShowerTemplate* muonTempl) {
  m_gtMuonShowerTemplate = muonTempl;
}

/// Set the pointer to GTL
void l1t::MuonShowerCondition::setGtGTL(const GlobalBoard* ptrGTL) { m_gtGTL = ptrGTL; }

// Try all object permutations
const bool l1t::MuonShowerCondition::evaluateCondition(const int bxEval) const {
  bool condResult = false;

  /* Number of trigger objects in the condition:
  // it is always 1 because at the uGT there is only one shower object per BX (that can be of several types).
  // See DN2020_033_v4 (sections 7.5 and 7.6) for reference
  */
  int nObjInCond = m_gtMuonShowerTemplate->nrObjects();

  const BXVector<std::shared_ptr<l1t::MuonShower>>* candVec = m_gtGTL->getCandL1MuShower();

  // Look at objects in BX = BX + relativeBX
  int useBx = bxEval + m_gtMuonShowerTemplate->condRelativeBx();
  LogDebug("MuonShowerCondition") << "Considering BX " << useBx << std::endl;

  // Fail condition if attempting to get BX outside of range
  if ((useBx < candVec->getFirstBX()) || (useBx > candVec->getLastBX())) {
    return false;
  }

  // Store the indices of the shower objects from the combination evaluated in the condition
  SingleCombInCond objectsInComb;
  objectsInComb.reserve(nObjInCond);

  // Clear the m_combinationsInCond vector
  combinationsInCond().clear();
  // Clear the indices in the combination
  objectsInComb.clear();

  /* If no candidates, no need to check further.
  // If there is a muon shower trigger, the size of the candidates vector is always 4:
  // in fact, we have four muon shower objects created in the Global Board.
  */
  int numberObjects = candVec->size(useBx);
  if (numberObjects < 1) {
    return false;
  }

  std::vector<int> index(numberObjects);
  for (int i = 0; i < numberObjects; ++i) {
    index[i] = i;
  }

  // index is always zero, as they are global quantities (there is only one object)
  int indexObj = 0;

  bool passCondition = false;

  for (int i = 0; i < numberObjects; i++) {
    passCondition = checkObjectParameter(0, *(candVec->at(useBx, index[i])), index[i]);  //BLW Change for BXVector
    condResult |= passCondition;
    if (passCondition) {
      LogDebug("MuonShowerCondition")
          << "===> MuShowerCondition::evaluateCondition, PASS! This muon shower passed the condition." << std::endl;
      objectsInComb.push_back(indexObj);
    } else
      LogDebug("MuonShowerCondition")
          << "===> MuShowerCondition::evaluateCondition, FAIL! This muon shower failed the condition." << std::endl;
  }

  // if we get here all checks were successfull for this combination
  // set the general result for evaluateCondition to "true"
  (combinationsInCond()).push_back(objectsInComb);

  return condResult;
}

/**
 * checkObjectParameter - Check if the bit associated to the type of shower is set to 1
 *
 * @param iCondition The number of the condition.
 * @param cand The candidate to compare.
 * @return The result of the check on the condition (false if a condition does not exist)
 */

const bool l1t::MuonShowerCondition::checkObjectParameter(const int iCondition,
                                                          const l1t::MuonShower& cand,
                                                          const unsigned int index) const {
  // number of objects in condition
  int nObjInCond = m_gtMuonShowerTemplate->nrObjects();

  if (iCondition >= nObjInCond || iCondition < 0) {
    return false;
  }

  const MuonShowerTemplate::ObjectParameter objPar = (*(m_gtMuonShowerTemplate->objectParameter()))[iCondition];

  LogDebug("L1TGlobal") << "\n MuonShowerTemplate::ObjectParameter (utm objects, checking which condition is parsed): "
                        << std::hex << "\n\t MuonShower0 = 0x " << objPar.MuonShower0 << "\n\t MuonShower1 = 0x "
                        << objPar.MuonShower1 << "\n\t MuonShowerOutOfTime0 = 0x " << objPar.MuonShowerOutOfTime0
                        << "\n\t MuonShowerOutOfTime1 = 0x " << objPar.MuonShowerOutOfTime1 << std::endl;

  LogDebug("L1TGlobal") << "\n l1t::MuonShower (uGT emulator bits): "
                        << "\n\t MuonShower0: isOneNominalInTime() = " << cand.isOneNominalInTime()
                        << "\n\t MuonShower1: isOneTightInTime() = " << cand.isOneTightInTime()
                        << "\n\t MuonShowerOutOfTime0: musOutOfTime0() = " << cand.musOutOfTime0()
                        << "\n\t MuonShowerOutOfTime1: musOutOfTime1() = " << cand.musOutOfTime1() << std::endl;

  // Check oneNominalInTime
  if (cand.isOneNominalInTime() != objPar.MuonShower0) {
    LogDebug("L1TGlobal") << "\t\t MuonShower failed MuonShower0 requirement" << std::endl;
    return false;
  }
  // Check oneTightInTime
  if (cand.isOneTightInTime() != objPar.MuonShower1) {
    LogDebug("L1TGlobal") << "\t\t MuonShower failed MuonShower1 requirement" << std::endl;
    return false;
  }
  if (cand.musOutOfTime0() != objPar.MuonShowerOutOfTime0) {
    LogDebug("L1TGlobal") << "\t\t MuonShower failed MuonShowerOutOfTime0 requirement" << std::endl;
    return false;
  }
  if (cand.musOutOfTime1() != objPar.MuonShowerOutOfTime1) {
    LogDebug("L1TGlobal") << "\t\t MuonShower failed MuonShowerOutOfTime1 requirement" << std::endl;
    return false;
  }

  return true;
}

void l1t::MuonShowerCondition::print(std::ostream& myCout) const {
  m_gtMuonShowerTemplate->print(myCout);
  ConditionEvaluation::print(myCout);
}
