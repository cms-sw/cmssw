#ifndef CondFormats_L1TObjects_GlobalCorrelationTemplate_h
#define CondFormats_L1TObjects_GlobalCorrelationTemplate_h

/**
 * \class GlobalCorrelationTemplate
 *
 *
 * Description: L1 Global Trigger correlation template.
 * Includes spatial correlation for two objects of different type.
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
#include <iosfwd>

#include <boost/cstdint.hpp>

// user include files

//   base class
#include "CondFormats/L1TObjects/interface/GlobalCondition.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/GlobalDefinitions.h"
// forward declarations

// class declaration
class GlobalCorrelationTemplate : public GlobalCondition
{

public:

    /// constructor(s)
    ///   default
    GlobalCorrelationTemplate();

    ///   from condition name
    GlobalCorrelationTemplate(const std::string& );

    ///   from condition name, the category of first sub-condition, the category of the
    ///   second sub-condition, the index of first sub-condition in the cor* vector,
    ///   the index of second sub-condition in the cor* vector
    GlobalCorrelationTemplate(const std::string&,
            const l1t::GlobalConditionCategory&, const l1t::GlobalConditionCategory&,
            const int, const int);

    /// copy constructor
    GlobalCorrelationTemplate( const GlobalCorrelationTemplate& );

    /// destructor
    virtual ~GlobalCorrelationTemplate();

    /// assign operator
    GlobalCorrelationTemplate& operator= (const GlobalCorrelationTemplate&);

public:

    /// typedef for correlation parameters
    struct CorrelationParameter
    {
        std::string deltaEtaRange;

        std::string deltaPhiRange;
        unsigned int deltaPhiMaxbits;
    };


public:

    /// get / set the category of the two sub-conditions
    inline const l1t::GlobalConditionCategory cond0Category() const {
        return m_cond0Category;
    }

    inline const l1t::GlobalConditionCategory cond1Category() const {
        return m_cond1Category;
    }

    void setCond0Category(const l1t::GlobalConditionCategory&);
    void setCond1Category(const l1t::GlobalConditionCategory&);

    /// get / set the index of the two sub-conditions in the cor* vector from menu
    inline const int cond0Index() const {
        return m_cond0Index;
    }

    inline const int cond1Index() const {
        return m_cond1Index;
    }

    void setCond0Index(const int&);
    void setCond1Index(const int&);

    /// get / set correlation parameters

    inline const CorrelationParameter* correlationParameter() const
    {
        return &m_correlationParameter;
    }

    void setCorrelationParameter(const CorrelationParameter& corrParameter);


    /// print the condition
    virtual void print(std::ostream& myCout) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const GlobalCorrelationTemplate&);


private:

    /// copy function for copy constructor and operator=
    void copy( const GlobalCorrelationTemplate& cp);


private:

    l1t::GlobalConditionCategory m_cond0Category;
    l1t::GlobalConditionCategory m_cond1Category;
    int m_cond0Index;
    int m_cond1Index;
    CorrelationParameter m_correlationParameter;

};

#endif
