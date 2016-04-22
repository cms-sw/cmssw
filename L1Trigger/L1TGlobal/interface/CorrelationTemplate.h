#ifndef L1Trigger_L1TGlobal_CorrelationTemplate_h
#define L1Trigger_L1TGlobal_CorrelationTemplate_h

/**
 * \class CorrelationTemplate
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
#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"

#include "L1Trigger/L1TGlobal/interface/GlobalDefinitions.h"
// forward declarations

// class declaration
class CorrelationTemplate : public GlobalCondition
{

public:

    /// constructor(s)
    ///   default
    CorrelationTemplate();

    ///   from condition name
    CorrelationTemplate(const std::string& );

    ///   from condition name, the category of first sub-condition, the category of the
    ///   second sub-condition, the index of first sub-condition in the cor* vector,
    ///   the index of second sub-condition in the cor* vector
    CorrelationTemplate(const std::string&,
            const l1t::GtConditionCategory&, const l1t::GtConditionCategory&,
            const int, const int);

    /// copy constructor
    CorrelationTemplate( const CorrelationTemplate& );

    /// destructor
    virtual ~CorrelationTemplate();

    /// assign operator
    CorrelationTemplate& operator= (const CorrelationTemplate&);

public:

    /// typedef for correlation parameters
    struct CorrelationParameter
    {
	
	//Cut values in hardware
	long long minEtaCutValue;
	long long maxEtaCutValue; 
	unsigned int precEtaCut;

	long long minPhiCutValue;
	long long maxPhiCutValue; 
	unsigned int precPhiCut;

	long long minDRCutValue;
	long long maxDRCutValue;
	unsigned int precDRCut; 

	long long minMassCutValue;
	long long maxMassCutValue;
	unsigned int precMassCut; 

        //Requirement on charge of legs (currently only Mu-Mu).	
	unsigned int chargeCorrelation;

	int corrCutType;

    };


public:

    /// get / set the category of the two sub-conditions
    inline const l1t::GtConditionCategory cond0Category() const {
        return m_cond0Category;
    }

    inline const l1t::GtConditionCategory cond1Category() const {
        return m_cond1Category;
    }

    void setCond0Category(const l1t::GtConditionCategory&);
    void setCond1Category(const l1t::GtConditionCategory&);

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
    friend std::ostream& operator<<(std::ostream&, const CorrelationTemplate&);


private:

    /// copy function for copy constructor and operator=
    void copy( const CorrelationTemplate& cp);


private:

    l1t::GtConditionCategory m_cond0Category;
    l1t::GtConditionCategory m_cond1Category;
    int m_cond0Index;
    int m_cond1Index;
    CorrelationParameter m_correlationParameter;

};

#endif
