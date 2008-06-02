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

    /// get / set the trigger menu name
    inline const std::string& gtTriggerMenuName() const {
        return m_triggerMenuName;
    }

    void setGtTriggerMenuName(const std::string&);

    /// get / set the condition maps
    inline const std::vector<ConditionMap>& gtConditionMap() const {
        return m_conditionMap;
    }

    void setGtConditionMap(const std::vector<ConditionMap>&);

    /// get / set the algorithm map
    inline const AlgorithmMap& gtAlgorithmMap() const {
        return m_algorithmMap;
    }

    void setGtAlgorithmMap(const AlgorithmMap&);

    /// print the trigger menu
    /// allow various verbosity levels
    void print(std::ostream&, int&) const;
    
public:
    
    /// get the result for algorithm with name algName
    /// use directly the format of decisionWord (no typedef) 
    const bool gtAlgorithmResult(const std::string& algName,
            const std::vector<bool>& decWord) const;
    

private:

    /// menu name 
    std::string m_triggerMenuName;

    /// map containing the conditions (per condition chip)
    std::vector<ConditionMap> m_conditionMap;

    /// map containing the algorithms (global map)
    AlgorithmMap m_algorithmMap;

};

#endif /*CondFormats_L1TObjects_L1GtTriggerMenu_h*/
