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
 * $Date:$
 * $Revision:$
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
    /// from algorithm algoNameVal
    const CombinationsInCond* getCombinationsInCond(
        std::string algoNameVal, std::string condNameVal);

private:

    std::vector<L1GlobalTriggerObjectMap> m_GtObjectMap;

};

#endif /* L1GlobalTrigger_L1GlobalTriggerObjectMapRecord_h */
