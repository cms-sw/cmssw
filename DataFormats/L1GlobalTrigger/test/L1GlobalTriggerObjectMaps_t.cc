
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMaps.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include <vector>
#include <iostream>
#include <cassert>

int main() {
  L1GlobalTriggerObjectMaps objMaps1;
  L1GlobalTriggerObjectMaps objMaps2;

  objMaps1.pushBackAlgorithm(0, 1, true);

  objMaps1.pushBackAlgorithm(0, 2, false);
  objMaps1.pushBackCondition(0, 3, true);
  objMaps1.pushBackObjectIndex(2);
  objMaps1.pushBackObjectIndex(0);
  objMaps1.pushBackObjectIndex(1);
  objMaps1.pushBackObjectIndex(21);
  objMaps1.pushBackObjectIndex(11);
  objMaps1.pushBackObjectIndex(31);
  objMaps1.pushBackCondition(6, 0, false);

  objMaps1.pushBackAlgorithm(2, 3, true);
  objMaps1.pushBackCondition(6, 1, true);
  objMaps1.pushBackObjectIndex(11);

  objMaps1.pushBackAlgorithm(3, 5, true);

  objMaps1.consistencyCheck();

  edm::ParameterSet pset;
  pset.addParameter<int>("blah", 1);
  pset.registerIt();
  edm::ParameterSetID psetID = pset.id();
  objMaps1.setNamesParameterSetID(psetID);

  swap(objMaps1, objMaps2);

  assert(psetID == objMaps2.namesParameterSetID());

  std::vector<int> algoBitNumbers;
  algoBitNumbers.push_back(11);  // should get erased
  objMaps1.getAlgorithmBitNumbers(algoBitNumbers);
  std::vector<int> expected;  // empty at this point
  assert(expected == algoBitNumbers);

  objMaps2.getAlgorithmBitNumbers(algoBitNumbers);
  expected.push_back(1);
  expected.push_back(2);
  expected.push_back(3);
  expected.push_back(5);
  assert(expected == algoBitNumbers);

  assert(objMaps2.algorithmResult(1) == true);
  assert(objMaps2.algorithmResult(2) == false);
  assert(objMaps2.algorithmResult(3) == true);

  L1GlobalTriggerObjectMaps::AlgorithmResult ar1;
  assert(ar1.startIndexOfConditions() == 0);
  assert(ar1.algorithmBitNumber() == 0);
  assert(ar1.algorithmResult() == false);

  L1GlobalTriggerObjectMaps::AlgorithmResult ar2(5, 1, true);
  assert(ar2.startIndexOfConditions() == 5);
  assert(ar2.algorithmBitNumber() == 1);
  assert(ar2.algorithmResult() == true);
  L1GlobalTriggerObjectMaps::AlgorithmResult ar3(6, 1, true);
  assert(ar1 < ar2);
  assert(!(ar2 < ar3));

  L1GlobalTriggerObjectMaps::ConditionResult cr1;
  assert(cr1.startIndexOfCombinations() == 0);
  assert(cr1.nObjectsPerCombination() == 0);
  assert(cr1.conditionResult() == false);

  L1GlobalTriggerObjectMaps::ConditionResult cr2(10, 11, true);
  assert(cr2.startIndexOfCombinations() == 10);
  assert(cr2.nObjectsPerCombination() == 11);
  assert(cr2.conditionResult() == true);

  assert(objMaps2.getNumberOfConditions(1) == 0);
  assert(objMaps2.getNumberOfConditions(2) == 2);
  assert(objMaps2.getNumberOfConditions(3) == 1);
  assert(objMaps2.getNumberOfConditions(5) == 0);

  std::vector<L1GtLogicParser::OperandToken> operandTokenVector;
  objMaps2.updateOperandTokenVector(1, operandTokenVector);

  L1GtLogicParser::OperandToken token0;
  token0.tokenName = "car";
  token0.tokenNumber = 5;
  token0.tokenResult = false;
  operandTokenVector.push_back(token0);

  L1GtLogicParser::OperandToken token1;
  token1.tokenName = "car";
  token1.tokenNumber = 5;
  token1.tokenResult = true;
  operandTokenVector.push_back(token1);

  objMaps2.updateOperandTokenVector(2, operandTokenVector);

  assert(operandTokenVector.at(0).tokenResult == true);
  assert(operandTokenVector.at(1).tokenResult == false);

  L1GlobalTriggerObjectMaps::ConditionsInAlgorithm conditions = objMaps2.getConditionsInAlgorithm(2);

  assert(conditions.nConditions() == 2);
  assert(conditions.getConditionResult(0) == true);
  assert(conditions.getConditionResult(1) == false);

  L1GlobalTriggerObjectMaps::CombinationsInCondition combinations = objMaps2.getCombinationsInCondition(2, 0);

  assert(combinations.nCombinations() == 2);
  assert(combinations.nObjectsPerCombination() == 3);
  assert(combinations.getObjectIndex(0, 0) == 2);
  assert(combinations.getObjectIndex(0, 1) == 0);
  assert(combinations.getObjectIndex(0, 2) == 1);
  assert(combinations.getObjectIndex(1, 0) == 21);
  assert(combinations.getObjectIndex(1, 1) == 11);
  assert(combinations.getObjectIndex(1, 2) == 31);

  L1GlobalTriggerObjectMaps::CombinationsInCondition combinations1 = objMaps2.getCombinationsInCondition(3, 0);

  assert(combinations1.nCombinations() == 1);
  assert(combinations1.nObjectsPerCombination() == 1);
  assert(combinations1.getObjectIndex(0, 0) == 11);

  L1GlobalTriggerObjectMaps::CombinationsInCondition combinations2 = objMaps2.getCombinationsInCondition(2, 1);

  assert(combinations2.nCombinations() == 0);
  assert(combinations2.nObjectsPerCombination() == 0);

  return 0;
}
