#ifndef CondFormats_L1TObjects_GlobalCondition_h
#define CondFormats_L1TObjects_GlobalCondition_h

/**
 * \class GlobalCondition
 *
 *
 * Description: base class for L1 Global Trigger object templates (condition).
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

#include <iostream>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "CondFormats/L1TObjects/interface/GlobalDefinitions.h"

// forward declarations

// class declaration
class GlobalCondition
{

public:

    /// constructor(s)
    ///
    GlobalCondition();

    ///   constructor from condition name
    GlobalCondition(const std::string& );

    ///   constructor from condition name, category and type
    GlobalCondition(const std::string&,
                  const l1t::GlobalConditionCategory&, const l1t::GlobalConditionType& );

    /// destructor
    virtual ~GlobalCondition();

public:

    /// get / set condition name
    inline const std::string& condName() const
    {
        return m_condName;
    }

    inline void setCondName(const std::string& cName)
    {
        m_condName = cName;
    }

    /// get / set the category of the condition
    inline const l1t::GlobalConditionCategory& condCategory() const
    {
        return m_condCategory;
    }

    inline void setCondCategory(const l1t::GlobalConditionCategory& cCategory)
    {
        m_condCategory = cCategory;
    }

    /// get / set the type of the condition (1s, etc)
    inline const l1t::GlobalConditionType& condType() const
    {
        return m_condType;
    }

    inline void setCondType(const l1t::GlobalConditionType& cType)
    {
        m_condType = cType;
    }

    /// get / set the trigger object type(s) in the condition
    inline const std::vector<L1GtObject>& objectType() const
    {
        return m_objectType;
    }

    inline void setObjectType(const std::vector<L1GtObject>& objType)
    {
        m_objectType = objType;
    }

    /// get / set condition GEq flag
    inline const bool condGEq() const
    {
        return m_condGEq;
    }

    inline void setCondGEq(const bool& cGEq)
    {
        m_condGEq = cGEq;
    }

    /// get / set the condition-chip number the condition is located on
    inline const int& condChipNr() const
    {
        return m_condChipNr;
    }

    inline void setCondChipNr(const int& cChipNr)
    {
        m_condChipNr = cChipNr;
    }

    /// get / set the condition relative bx
    inline const int& condRelativeBx() const
    {
        return m_condRelativeBx;
    }

    inline void setCondRelativeBx(const int& cRelativeBx)
    {
        m_condRelativeBx = cRelativeBx;
    }


public:

    /// get number of trigger objects
    const int nrObjects() const;

    /// get logic flag for conditions, same type of trigger objects,
    /// and with spatial correlations
    const bool wsc() const;

    /// get logic flag for conditions, different type of trigger objects,
    /// and with spatial correlations
    const bool corr() const;

    /// print condition
    virtual void print(std::ostream& myCout) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const GlobalCondition&);

protected:

    /// the name of the condition
    std::string m_condName;

    /// the category of the condition
    l1t::GlobalConditionCategory m_condCategory;

    /// the type of the condition (1s, etc)
    l1t::GlobalConditionType m_condType;

    /// the trigger object type(s)
    std::vector<L1GtObject> m_objectType;

    /// the operator used for the condition (>=, =): true for >=
    bool m_condGEq;

    /// condition is located on condition chip m_condChipNr
    int m_condChipNr;

    // Relative bunch crossing offset for input data.
    int m_condRelativeBx;
};

#endif /*CondFormats_L1TObjects_GlobalCondition_h*/
