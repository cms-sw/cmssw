#ifndef L1GlobalTrigger_L1GlobalTriggerObjectMapRecord_h
#define L1GlobalTrigger_L1GlobalTriggerObjectMapRecord_h

/**
 * \class L1GlobalTriggerObjectMapRecord
 * 
 * 
 * 
 * Description: map trigger objects to algorithms and conditions 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
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
class L1GlobalTriggerObjectMapRecord
{

public:

    /// constructor(s)
    L1GlobalTriggerObjectMapRecord();

    /// destructor
    virtual ~L1GlobalTriggerObjectMapRecord();

public:

    /// return all the combinations passing the requirements imposed in condition condNameVal
    /// from algorithm with name algoNameVal
    const CombinationsInCond* getCombinationsInCond(
        std::string algoNameVal, std::string condNameVal) const;

    /// return all the combinations passing the requirements imposed in condition condNameVal
    /// from algorithm with bit number algoBitNumberVal
    const CombinationsInCond* getCombinationsInCond(
        int algoBitNumberVal, std::string condNameVal) const;

    /// return the result for the condition condNameVal
    /// from algorithm with name algoNameVal
    const bool getConditionResult(std::string algoNameVal, std::string condNameVal) const;

    /// return the result for the condition condNameVal
    /// from algorithm with bit number algoBitNumberVal
    const bool getConditionResult(int algoBitNumberVal, std::string condNameVal) const;

public:

    /// get / set the vector of object maps
    inline const std::vector<L1GlobalTriggerObjectMap>& gtObjectMap() const
    {
        return m_gtObjectMap;
    }

    void setGtObjectMap(const std::vector<L1GlobalTriggerObjectMap>& gtObjectMapValue)
    {
        m_gtObjectMap = gtObjectMapValue;
    }

private:

    std::vector<L1GlobalTriggerObjectMap> m_gtObjectMap;

};

#endif /* L1GlobalTrigger_L1GlobalTriggerObjectMapRecord_h */
