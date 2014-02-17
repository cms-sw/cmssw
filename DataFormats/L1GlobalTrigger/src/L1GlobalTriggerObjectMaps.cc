/**
 * \class L1GlobalTriggerObjectMapRecord
 * 
 * 
 * Description: see header file.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * \author: W. David Dagenhart
 * 
 * $Date: 2012/03/02 21:46:28 $
 * $Revision: 1.1 $
 *
 */

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMaps.h"

#include <algorithm>
#include <limits>

#include "FWCore/Utilities/interface/Exception.h"

void L1GlobalTriggerObjectMaps::swap(L1GlobalTriggerObjectMaps& rh) {
  m_algorithmResults.swap(rh.m_algorithmResults);
  m_conditionResults.swap(rh.m_conditionResults);
  m_combinations.swap(rh.m_combinations);
  m_namesParameterSetID.swap(rh.m_namesParameterSetID);
}

bool L1GlobalTriggerObjectMaps::algorithmExists(int algorithmBitNumber) const {
  std::vector<AlgorithmResult>::const_iterator i =
    std::lower_bound(m_algorithmResults.begin(),
                     m_algorithmResults.end(),
                     AlgorithmResult(0, algorithmBitNumber, false));
  if (i == m_algorithmResults.end() || i->algorithmBitNumber() != algorithmBitNumber) {
    return false;
  }
  return true;
}

bool L1GlobalTriggerObjectMaps::algorithmResult(int algorithmBitNumber) const {
  std::vector<AlgorithmResult>::const_iterator i =
    std::lower_bound(m_algorithmResults.begin(),
                     m_algorithmResults.end(),
                     AlgorithmResult(0, algorithmBitNumber, false));
  if (i == m_algorithmResults.end() || i->algorithmBitNumber() != algorithmBitNumber) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "algorithmBitNumber not found";
    ex.addContext("Calling L1GlobalTriggerObjectMaps::algorithmResult");
    throw ex;
  }
  return i->algorithmResult();
}

void L1GlobalTriggerObjectMaps::
updateOperandTokenVector(int algorithmBitNumber,
                         std::vector<L1GtLogicParser::OperandToken>& operandTokenVector) const {
  unsigned startIndex = 0;
  unsigned endIndex = 0;
  getStartEndIndex(algorithmBitNumber, startIndex, endIndex);

  unsigned length = endIndex - startIndex;
  if (length != operandTokenVector.size()) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "operand token vector size does not match number of conditions";
    ex.addContext("Calling L1GlobalTriggerObjectMaps::updateOperandTokenVector");
    throw ex;
  }

  for (unsigned i = 0; i < length; ++i) {
    operandTokenVector[i].tokenResult = m_conditionResults[startIndex + i].conditionResult();
  }
}

void L1GlobalTriggerObjectMaps::
getAlgorithmBitNumbers(std::vector<int>& algorithmBitNumbers) const {
  algorithmBitNumbers.clear();
  for (std::vector<AlgorithmResult>::const_iterator i = m_algorithmResults.begin(),
       iEnd = m_algorithmResults.end(); i != iEnd; ++i) {
    algorithmBitNumbers.push_back(i->algorithmBitNumber());
  }
}

unsigned L1GlobalTriggerObjectMaps::
getNumberOfConditions(int algorithmBitNumber) const {

  unsigned startIndex = 0;
  unsigned endIndex = 0;
  getStartEndIndex(algorithmBitNumber, startIndex, endIndex);
  return endIndex - startIndex;
}

L1GlobalTriggerObjectMaps::ConditionsInAlgorithm L1GlobalTriggerObjectMaps::
getConditionsInAlgorithm(int algorithmBitNumber) const {
  unsigned startIndex = 0;
  unsigned endIndex = 0;
  getStartEndIndex(algorithmBitNumber, startIndex, endIndex);
  return ConditionsInAlgorithm(&m_conditionResults[startIndex], endIndex - startIndex);
}

L1GlobalTriggerObjectMaps::CombinationsInCondition L1GlobalTriggerObjectMaps::
getCombinationsInCondition(int algorithmBitNumber,
                           unsigned conditionNumber) const {
  unsigned startIndex = 0;
  unsigned endIndex = 0;
  getStartEndIndex(algorithmBitNumber, startIndex, endIndex);

  if (endIndex <= startIndex + conditionNumber) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "Condition number is out of range";
    ex.addContext("Calling L1GlobalTriggerObjectMaps::getCombinationsInCondition");
    throw ex;
  }

  unsigned endObjectIndex = m_combinations.size();
  unsigned nextConditionIndex = startIndex + conditionNumber + 1U;
  if (nextConditionIndex < m_conditionResults.size()) {
    endObjectIndex = m_conditionResults[nextConditionIndex].startIndexOfCombinations();
  }
  unsigned beginObjectIndex = m_conditionResults[startIndex + conditionNumber].startIndexOfCombinations();
  unsigned short nObjectsPerCombination = m_conditionResults[startIndex + conditionNumber].nObjectsPerCombination();

  if (endObjectIndex == beginObjectIndex) {
    return CombinationsInCondition(0, 0, 0);
  }
  if (endObjectIndex < beginObjectIndex ||
      m_combinations.size() < endObjectIndex ||
      nObjectsPerCombination == 0 ||
      (endObjectIndex - beginObjectIndex) % nObjectsPerCombination != 0) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "Indexes to combinations are invalid";
    ex.addContext("Calling L1GlobalTriggerObjectMaps::getCombinationsInCondition");
    throw ex;
  }
  return CombinationsInCondition(&m_combinations[beginObjectIndex],
                                 (endObjectIndex - beginObjectIndex) / nObjectsPerCombination,
                                  nObjectsPerCombination);
}

void L1GlobalTriggerObjectMaps::
reserveForAlgorithms(unsigned n) {
  m_algorithmResults.reserve(n);
}

void L1GlobalTriggerObjectMaps::
pushBackAlgorithm(unsigned startIndexOfConditions,
                  int algorithmBitNumber,
                  bool algorithmResult) {
  m_algorithmResults.push_back(AlgorithmResult(startIndexOfConditions,
                                               algorithmBitNumber,
                                               algorithmResult));
}

void L1GlobalTriggerObjectMaps::consistencyCheck() const {
  // None of these checks should ever fail unless there
  // is a bug in the code filling this object
  for (std::vector<AlgorithmResult>::const_iterator i = m_algorithmResults.begin(),
         iEnd = m_algorithmResults.end(); i != iEnd; ++i) {
    std::vector<AlgorithmResult>::const_iterator j = i;
    ++j;
    if (j != iEnd && !(*i < *j)) {
      cms::Exception ex("L1GlobalTrigger");
      ex << "AlgorithmResults should be sorted in increasing order of bit number with no duplicates. It is not.";
      ex.addContext("Calling L1GlobalTriggerObjectMaps::consistencyCheck");
      throw ex;      
    }
    unsigned endIndex = 
      (j != iEnd) ? j->startIndexOfConditions() : m_conditionResults.size();

    if (endIndex < i->startIndexOfConditions()) {
      cms::Exception ex("L1GlobalTrigger");
      ex << "startIndexOfConditions decreases or exceeds the size of m_conditionResults";
      ex.addContext("Calling L1GlobalTriggerObjectMaps::consistencyCheck");
      throw ex;
    }
  }
  for (std::vector<ConditionResult>::const_iterator i = m_conditionResults.begin(),
         iEnd = m_conditionResults.end(); i != iEnd; ++i) {
    std::vector<ConditionResult>::const_iterator j = i;
    ++j;
    unsigned endIndex = 
      (j != iEnd) ? j->startIndexOfCombinations() : m_combinations.size();

    if (endIndex < i->startIndexOfCombinations()) {
      cms::Exception ex("L1GlobalTrigger");
      ex << "startIndexOfCombinations decreases or exceeds the size of m_conditionResults";
      ex.addContext("Calling L1GlobalTriggerObjectMaps::consistencyCheck");
      throw ex;
    }
    unsigned length = endIndex - i->startIndexOfCombinations();
    if (length == 0U) {
      if (i->nObjectsPerCombination() != 0U) {
        cms::Exception ex("L1GlobalTrigger");
        ex << "Length is zero and nObjectsInCombination is not zero";
        ex.addContext("Calling L1GlobalTriggerObjectMaps::consistencyCheck");
        throw ex;
      }
    } else {
      if (i->nObjectsPerCombination() == 0 || length % i->nObjectsPerCombination() != 0) {
        cms::Exception ex("L1GlobalTrigger");
        ex << "Size indicated by startIndexOfCombinations is not a multiple of nObjectsInCombination";
        ex.addContext("Calling L1GlobalTriggerObjectMaps::consistencyCheck");
        throw ex;
      }
    }
  }
}

void L1GlobalTriggerObjectMaps::
reserveForConditions(unsigned n) {
  m_conditionResults.reserve(n);
}

void L1GlobalTriggerObjectMaps::
pushBackCondition(unsigned startIndexOfCombinations,
                  unsigned short nObjectsPerCombination,
                  bool conditionResult) {
  m_conditionResults.push_back(ConditionResult(startIndexOfCombinations,
                                               nObjectsPerCombination,
                                               conditionResult));
}

void L1GlobalTriggerObjectMaps::
reserveForObjectIndexes(unsigned n) {
  m_combinations.reserve(n);
}

void L1GlobalTriggerObjectMaps::
pushBackObjectIndex(unsigned char objectIndex) {
  m_combinations.push_back(objectIndex);
}

void L1GlobalTriggerObjectMaps::
setNamesParameterSetID(edm::ParameterSetID const& psetID) {
  m_namesParameterSetID = psetID;
}

L1GlobalTriggerObjectMaps::AlgorithmResult::
AlgorithmResult() :
  m_startIndexOfConditions(0),
  m_algorithmBitNumber(0),
  m_algorithmResult(false) {
}

L1GlobalTriggerObjectMaps::AlgorithmResult::
AlgorithmResult(unsigned startIndexOfConditions,
                int algorithmBitNumber,
                bool algorithmResult) :
  m_startIndexOfConditions(startIndexOfConditions),
  m_algorithmResult(algorithmResult) {

  // We made the decision to try to save space in the data format
  // and fit this object into 8 bytes by making the persistent
  // algorithmBitNumber a short. This creates something very
  // ugly below.  In practice the range should never be exceeded.
  // In fact it is currently always supposed to be less than 128.
  // I hope this never comes back to haunt us for some unexpected reason.
  // I cringe when I look at it, but cannot think of any practical
  // harm ... It is probably a real bug if anyone ever
  // tries to shove a big int into here.
  if (algorithmBitNumber < std::numeric_limits<short>::min() ||
      algorithmBitNumber > std::numeric_limits<short>::max()) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "algorithmBitNumber out of range of a short int";
    ex.addContext("Calling L1GlobalTriggerObjectMaps::AlgorithmResult::AlgorithmResult");
    throw ex;    
  }
  m_algorithmBitNumber = static_cast<short>(algorithmBitNumber);
}

L1GlobalTriggerObjectMaps::ConditionResult::
ConditionResult() :
  m_startIndexOfCombinations(0),
  m_nObjectsPerCombination(0),
  m_conditionResult(false) {
}

L1GlobalTriggerObjectMaps::ConditionResult::
ConditionResult(unsigned startIndexOfCombinations,
                unsigned short nObjectsPerCombination,
                bool conditionResult) :
  m_startIndexOfCombinations(startIndexOfCombinations),
  m_nObjectsPerCombination(nObjectsPerCombination),
  m_conditionResult(conditionResult) {
}

L1GlobalTriggerObjectMaps::ConditionsInAlgorithm::
ConditionsInAlgorithm(ConditionResult const* conditionResults,
                      unsigned nConditions) :
  m_conditionResults(conditionResults),
  m_nConditions(nConditions) {
}

bool L1GlobalTriggerObjectMaps::ConditionsInAlgorithm::
getConditionResult(unsigned condition) const {
  if (condition >= m_nConditions) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "argument out of range";
    ex.addContext("Calling L1GlobalTriggerObjectMaps::ConditionsInAlgorithm::getConditionResult");
    throw ex;
  }
  return (m_conditionResults + condition)->conditionResult();
}

L1GlobalTriggerObjectMaps::CombinationsInCondition::
CombinationsInCondition(unsigned char const* startOfObjectIndexes,
                        unsigned nCombinations,
                        unsigned short nObjectsPerCombination) :
  m_startOfObjectIndexes(startOfObjectIndexes),
  m_nCombinations(nCombinations),
  m_nObjectsPerCombination(nObjectsPerCombination) {
}

unsigned char L1GlobalTriggerObjectMaps::CombinationsInCondition::
getObjectIndex(unsigned combination,
               unsigned object) const {
  if (combination >= m_nCombinations ||
      object >=  m_nObjectsPerCombination) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "arguments out of range";
    ex.addContext("Calling L1GlobalTriggerObjectMaps::CombinationsInCondition::getObjectIndex");
    throw ex;
  }
  return m_startOfObjectIndexes[combination * m_nObjectsPerCombination + object];
}

void L1GlobalTriggerObjectMaps::
getStartEndIndex(int algorithmBitNumber, unsigned& startIndex, unsigned& endIndex) const {
  std::vector<AlgorithmResult>::const_iterator iAlgo =
    std::lower_bound(m_algorithmResults.begin(),
                     m_algorithmResults.end(),
                     AlgorithmResult(0, algorithmBitNumber, false));

  if (iAlgo == m_algorithmResults.end() || iAlgo->algorithmBitNumber() != algorithmBitNumber) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "algorithmBitNumber not found";
    ex.addContext("Calling L1GlobalTriggerObjectMaps::getStartEndIndex");
    throw ex;
  }

  startIndex = iAlgo->startIndexOfConditions();
  ++iAlgo;
  endIndex = (iAlgo != m_algorithmResults.end()) ? 
             iAlgo->startIndexOfConditions() : m_conditionResults.size();

  if (endIndex < startIndex || m_conditionResults.size() < endIndex) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "index out of order or out of range";
    ex.addContext("Calling L1GlobalTriggerObjectMaps::getStartEndIndex");
    throw ex;
  }
}
