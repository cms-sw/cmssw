#ifndef CondFormats_L1TObjects_L1GtTriggerMenu_h
#define CondFormats_L1TObjects_L1GtTriggerMenu_h

/**
 * \class L1GtTriggerMenu
 * 
 * 
 * Description: L1 trigger menu.  
 *
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
#include <map>

#include <ostream>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

// forward declarations
class L1GtCondition;
class L1GtAlgorithm;

// class declaration
class L1GtTriggerMenu
{

public:

    // constructor
    L1GtTriggerMenu();

    // destructor
    virtual ~L1GtTriggerMenu();

public:

    /// print the trigger menu
    /// allow various verbosity levels
    void print(std::ostream&, int&) const;

private:

    /// clear the maps
    void clearMaps();

    /// insertConditionIntoMap - safe insert of condition into condition map.
    /// if the condition name already exists, do not insert it and return false
    bool insertConditionIntoMap(L1GtCondition* cond, int chipNr,
                                std::ostream& myCout);

    /// insert algo into a map
    bool insertAlgoIntoMap(L1GtAlgorithm* algo,
                           AlgorithmsMap* insertMap, ConditionsMap* operandMap,
                           int chipNr, std::ostream& myCout);

private:

    /// map containing the conditions (per condition chip)
    std::vector<ConditionsMap> m_conditionsMap;

    /// map containing the algorithms (global map)
    AlgorithmsMap m_algorithmsMap;

};

#endif /*CondFormats_L1TObjects_L1GtTriggerMenu_h*/
