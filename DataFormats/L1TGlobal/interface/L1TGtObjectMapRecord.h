#ifndef L1GlobalTrigger_L1TGtObjectMapRecord_h
#define L1GlobalTrigger_L1TGtObjectMapRecord_h

/**
 * \class L1TGtObjectMapRecord
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
#include "DataFormats/L1TGlobal/interface/L1TGtObjectMapFwd.h"
#include "DataFormats/L1TGlobal/interface/L1TGtObjectMap.h"

// forward declarations

// class declaration
class L1TGtObjectMapRecord
{

public:

    /// constructor(s)
  L1TGtObjectMapRecord() {}

  /// destructor
  ~L1TGtObjectMapRecord() {}

  void swap(L1TGtObjectMapRecord & rh) {
    m_gtObjectMap.swap(rh.m_gtObjectMap);
  }

public:

    /// return the object map for the algorithm algoNameVal
    const L1TGtObjectMap* getObjectMap(const std::string& algoNameVal) const;
    
    /// return the object map for the algorithm with bit number const int algoBitNumberVal
    const L1TGtObjectMap* getObjectMap(const int algoBitNumberVal) const;

    /// return all the combinations passing the requirements imposed in condition condNameVal
    /// from algorithm with name algoNameVal
    const CombinationsInCond* getCombinationsInCond(
        const std::string& algoNameVal, const std::string& condNameVal) const;

    /// return all the combinations passing the requirements imposed in condition condNameVal
    /// from algorithm with bit number algoBitNumberVal
    const CombinationsInCond* getCombinationsInCond(
        const int algoBitNumberVal, const std::string& condNameVal) const;

    /// return the result for the condition condNameVal
    /// from algorithm with name algoNameVal
    bool getConditionResult(const std::string& algoNameVal, const std::string& condNameVal) const;

    /// return the result for the condition condNameVal
    /// from algorithm with bit number algoBitNumberVal
    bool getConditionResult(const int algoBitNumberVal, const std::string& condNameVal) const;

public:

    /// get / set the vector of object maps
    inline const std::vector<L1TGtObjectMap>& gtObjectMap() const
    {
        return m_gtObjectMap;
    }

    inline void setGtObjectMap(const std::vector<L1TGtObjectMap>& gtObjectMapValue)
    {
        m_gtObjectMap = gtObjectMapValue;
    }

   inline void swapGtObjectMap(std::vector<L1TGtObjectMap>& gtObjectMapValue)
    {
      m_gtObjectMap.swap(gtObjectMapValue);
    }


private:

    std::vector<L1TGtObjectMap> m_gtObjectMap;

};

inline void swap( L1TGtObjectMapRecord & lh,  L1TGtObjectMapRecord& rh) {
  lh.swap(rh);
}

#endif /* L1GlobalTrigger_L1TGtObjectMapRecord_h */
