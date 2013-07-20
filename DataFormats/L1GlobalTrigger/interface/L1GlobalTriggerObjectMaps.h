#ifndef DataFormats_L1GlobalTrigger_L1GlobalTriggerObjectMaps_h
#define DataFormats_L1GlobalTrigger_L1GlobalTriggerObjectMaps_h

/**
 * \class L1GlobalTriggerObjectMaps
 * 
 * 
 * Description: map trigger objects to algorithms and conditions.  
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

#include <vector>

#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

class L1GlobalTriggerObjectMaps
{

public:

  L1GlobalTriggerObjectMaps() {}

  ~L1GlobalTriggerObjectMaps() {}

  void swap(L1GlobalTriggerObjectMaps& rh);

  /// Returns true if there is an entry for this algorithm bit number
  bool algorithmExists(int algorithmBitNumber) const;

  /// Returns whether an algorithm trigger passed or failed
  bool algorithmResult(int algorithmBitNumber) const;

  /// Update the condition result in the operandTokenVector.
  void updateOperandTokenVector(int algorithmBitNumber,
                                std::vector<L1GtLogicParser::OperandToken>& operandTokenVector) const;

  /// Fills the vector with all the algorithm bit numbers
  void getAlgorithmBitNumbers(std::vector<int>& algorithmBitNumbers) const;

  /// Number of conditions associated with an algorithm
  unsigned getNumberOfConditions(int algorithmBitNumber) const;

  class ConditionsInAlgorithm;
  class CombinationsInCondition;

  /// Return an object which has a function that returns the
  /// results of the conditions associated with an algorithm.
  ConditionsInAlgorithm getConditionsInAlgorithm(int algorithmBitNumber) const;

  /// Each condition can be satisfied by multiple combinations of L1
  /// objects.  The number, order, and type of objects associated with each
  /// condition is defined in the L1 Trigger Menu. The following function
  /// returns an object which has a function that returns the index into the
  /// L1 Object Collections of each object in each combination.
  CombinationsInCondition getCombinationsInCondition(int algorithmBitNumber,
                                                     unsigned conditionNumber) const;

  /// Returns the ID of the ParameterSet containing the algorithm names
  /// and condition names.
  edm::ParameterSetID const& namesParameterSetID() const { return m_namesParameterSetID; }

  // the rest are methods used to fill the data
  // and should only be used by the producer that
  // creates these and puts them into the event.
  // There are requirements on ordering and the
  // stored indexes that the code that calls these
  // functions is expected to carefully take care of.

  void reserveForAlgorithms(unsigned n);
  void pushBackAlgorithm(unsigned startIndexOfConditions,
                         int algorithmBitNumber,
                         bool algorithmResult);

  // This function should be called after filling the data in this object
  // using the pushBack* methods.
  void consistencyCheck() const;

  void reserveForConditions(unsigned n);
  void pushBackCondition(unsigned startIndexOfCombinations,
                         unsigned short nObjectsPerCombination,
                         bool conditionResult);

  void reserveForObjectIndexes(unsigned n);
  void pushBackObjectIndex(unsigned char objectIndex);

  void setNamesParameterSetID(edm::ParameterSetID const& psetID);

  class AlgorithmResult {
  public:
    AlgorithmResult();
    AlgorithmResult(unsigned startIndexOfConditions,
                    int algorithmBitNumber,
                    bool algorithmResult);
    unsigned startIndexOfConditions() const { return m_startIndexOfConditions; }
    short algorithmBitNumber() const { return m_algorithmBitNumber; }
    bool algorithmResult() const { return m_algorithmResult; }
    // The operator is used for searching in the std::vector<AlgorithmResult>
    bool operator<(AlgorithmResult const& right) const {
      return m_algorithmBitNumber < right.algorithmBitNumber();
    }
  private:
    unsigned m_startIndexOfConditions;
    short m_algorithmBitNumber;
    bool m_algorithmResult;
  };

  class ConditionResult {
  public:
    ConditionResult();
    ConditionResult(unsigned startIndexOfCombinations,
                    unsigned short nObjectsPerCombination,
                    bool conditionResult);
    unsigned startIndexOfCombinations() const { return m_startIndexOfCombinations; }
    unsigned short nObjectsPerCombination() const { return m_nObjectsPerCombination; }
    bool conditionResult() const { return m_conditionResult; }
  private:
    unsigned m_startIndexOfCombinations;
    unsigned short m_nObjectsPerCombination;
    bool m_conditionResult;
  };

  class ConditionsInAlgorithm {
  public:
    ConditionsInAlgorithm(ConditionResult const* conditionResults,
                          unsigned nConditions);
    unsigned nConditions() const { return m_nConditions; }
    bool getConditionResult(unsigned condition) const;

  private:
    ConditionResult const* m_conditionResults;
    unsigned m_nConditions;
  };

  class CombinationsInCondition {
  public:
    CombinationsInCondition(unsigned char const* startOfObjectIndexes,
                            unsigned nCombinations,
                            unsigned short nObjectsPerCombination);

    unsigned nCombinations() const { return m_nCombinations; }
    unsigned short nObjectsPerCombination() const { return m_nObjectsPerCombination; }
    unsigned char getObjectIndex(unsigned combination,
                                 unsigned object) const;
  private:
    unsigned char const* m_startOfObjectIndexes;
    unsigned m_nCombinations;
    unsigned short m_nObjectsPerCombination;
  };

private:

  void getStartEndIndex(int algorithmBitNumber, unsigned& startIndex, unsigned& endIndex) const;

  // data members
  // The vectors are sorted. All three vectors are in algorithmBitNumber
  // order. The second two vectors are sorted such that within an algorithm
  // the conditions appear in the same order as in the algorithm logical
  // expression. And the third is additionally sorted so each combination
  // is contiguous and within a combination the order is the same as the
  // type specification in the L1 Trigger Menu.
  std::vector<AlgorithmResult> m_algorithmResults;
  std::vector<ConditionResult> m_conditionResults;
  std::vector<unsigned char> m_combinations;
  edm::ParameterSetID m_namesParameterSetID;
};

inline void swap(L1GlobalTriggerObjectMaps& lh, L1GlobalTriggerObjectMaps& rh) {
  lh.swap(rh);
}

#endif
