#ifndef L1GlobalTrigger_L1GlobalTriggerObjectMapRecord_h
#define L1GlobalTrigger_L1GlobalTriggerObjectMapRecord_h

/**
 * \class L1GlobalTriggerObjectMapRecord
 * 
 * 
 * Description: map trigger objects to algorithms and conditions.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <string>
#include <vector>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

// forward declarations

// class declaration
class L1GlobalTriggerObjectMapRecord {
public:
  /// constructor(s)
  L1GlobalTriggerObjectMapRecord() {}

  /// destructor
  ~L1GlobalTriggerObjectMapRecord() {}

  void swap(L1GlobalTriggerObjectMapRecord& rh) { m_gtObjectMap.swap(rh.m_gtObjectMap); }

public:
  /// return the object map for the algorithm algoNameVal
  const L1GlobalTriggerObjectMap* getObjectMap(const std::string& algoNameVal) const;

  /// return the object map for the algorithm with bit number const int algoBitNumberVal
  const L1GlobalTriggerObjectMap* getObjectMap(const int algoBitNumberVal) const;

  /// return all the combinations passing the requirements imposed in condition condNameVal
  /// from algorithm with name algoNameVal
  const CombinationsInCond* getCombinationsInCond(const std::string& algoNameVal, const std::string& condNameVal) const;

  /// return all the combinations passing the requirements imposed in condition condNameVal
  /// from algorithm with bit number algoBitNumberVal
  const CombinationsInCond* getCombinationsInCond(const int algoBitNumberVal, const std::string& condNameVal) const;

  /// return the result for the condition condNameVal
  /// from algorithm with name algoNameVal
  bool getConditionResult(const std::string& algoNameVal, const std::string& condNameVal) const;

  /// return the result for the condition condNameVal
  /// from algorithm with bit number algoBitNumberVal
  bool getConditionResult(const int algoBitNumberVal, const std::string& condNameVal) const;

public:
  /// get / set the vector of object maps
  inline const std::vector<L1GlobalTriggerObjectMap>& gtObjectMap() const { return m_gtObjectMap; }

  inline void setGtObjectMap(const std::vector<L1GlobalTriggerObjectMap>& gtObjectMapValue) {
    m_gtObjectMap = gtObjectMapValue;
  }

  inline void swapGtObjectMap(std::vector<L1GlobalTriggerObjectMap>& gtObjectMapValue) {
    m_gtObjectMap.swap(gtObjectMapValue);
  }

private:
  std::vector<L1GlobalTriggerObjectMap> m_gtObjectMap;
};

inline void swap(L1GlobalTriggerObjectMapRecord& lh, L1GlobalTriggerObjectMapRecord& rh) { lh.swap(rh); }

#endif /* L1GlobalTrigger_L1GlobalTriggerObjectMapRecord_h */
