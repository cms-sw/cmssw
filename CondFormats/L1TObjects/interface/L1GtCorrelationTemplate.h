#ifndef CondFormats_L1TObjects_L1GtCorrelationTemplate_h
#define CondFormats_L1TObjects_L1GtCorrelationTemplate_h

/**
 * \class L1GtCorrelationTemplate
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
#include "CondFormats/L1TObjects/interface/L1GtCondition.h"

// forward declarations

// class declaration
class L1GtCorrelationTemplate : public L1GtCondition
{

public:

    /// constructor(s)
    ///   default
    L1GtCorrelationTemplate();

    ///   from condition name
    L1GtCorrelationTemplate(const std::string& );

    ///   from condition name and two existing conditions
    L1GtCorrelationTemplate(const std::string&, const L1GtCondition&, const L1GtCondition&);

    /// copy constructor
    L1GtCorrelationTemplate( const L1GtCorrelationTemplate& );

    /// destructor
    virtual ~L1GtCorrelationTemplate();

    /// assign operator
    L1GtCorrelationTemplate& operator= (const L1GtCorrelationTemplate&);

public:

    /// typedef for correlation parameters
    typedef struct
    {
        boost::uint64_t deltaEtaRange;

        boost::uint64_t deltaPhiRange;
        unsigned int deltaPhiMaxbits;
    }
    CorrelationParameter;


public:

    /// get / set methods for object conditions and correlation parameters

    inline const std::vector<L1GtCondition>* objectCondition() const
    {
        return &m_objectCondition;
    }

    inline const CorrelationParameter* correlationParameter() const
    {
        return &m_correlationParameter;
    }


    void setConditionParameter(const std::vector<L1GtCondition>& objCondition,
                               const CorrelationParameter& corrParameter);


    /// print the condition
    virtual void print(std::ostream& myCout) const;

private:

    /// copy function for copy constructor and operator=
    void copy( const L1GtCorrelationTemplate& cp);


private:

    /// variables containing the parameters
    std::vector<L1GtCondition> m_objectCondition;
    CorrelationParameter m_correlationParameter;

};

#endif
