#ifndef L1GlobalTrigger_L1TGtObjectMapRecord_h
#define L1GlobalTrigger_L1TGtObjectMapRecord_h

/**
 * \class GlobalObjectMapRecord
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
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapFwd.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMap.h"

// forward declarations

// class declaration
class GlobalObjectMapRecord
{

public:

    /// constructor(s)
  GlobalObjectMapRecord() {}

  /// destructor
  ~GlobalObjectMapRecord() {}

  void swap(GlobalObjectMapRecord & rh) {
    m_gtObjectMap.swap(rh.m_gtObjectMap);
  }

public:

    /// return the object map for the algorithm algoNameVal
    const GlobalObjectMap* getObjectMap(const std::string& algoNameVal) const;
    
    /// return the object map for the algorithm with bit number const int algoBitNumberVal
    const GlobalObjectMap* getObjectMap(const int algoBitNumberVal) const;

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
    inline const std::vector<GlobalObjectMap>& gtObjectMap() const
    {
        return m_gtObjectMap;
    }

    inline void setGtObjectMap(const std::vector<GlobalObjectMap>& gtObjectMapValue)
    {
        m_gtObjectMap = gtObjectMapValue;
    }

   inline void swapGtObjectMap(std::vector<GlobalObjectMap>& gtObjectMapValue)
    {
      m_gtObjectMap.swap(gtObjectMapValue);
    }


private:

    std::vector<GlobalObjectMap> m_gtObjectMap;

};

inline void swap( GlobalObjectMapRecord & lh,  GlobalObjectMapRecord& rh) {
  lh.swap(rh);
}

#endif /* L1GlobalTrigger_L1TGtObjectMapRecord_h */
