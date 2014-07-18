#ifndef L1Trigger_L1TGlobal_CaloTemplate_h
#define L1Trigger_L1TGlobal_CaloTemplate_h

/**
 * \class CaloTemplate
 *
 *
 * Description: L1 Global Trigger calo template.
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

// user include files

//   base class
#include "L1Trigger/L1TGlobal/interface/GtCondition.h"

// forward declarations

// class declaration
class CaloTemplate : public GtCondition
{

public:

    // constructor
    CaloTemplate();

    // constructor
    CaloTemplate(const std::string& );

    // constructor
    CaloTemplate(const std::string&, const l1t::GtConditionType& );

    // copy constructor
    CaloTemplate( const CaloTemplate& );

    // destructor
    virtual ~CaloTemplate();

    // assign operator
    CaloTemplate& operator= (const CaloTemplate&);

public:

    /// typedef for a single object template
    struct ObjectParameter
    {
      unsigned int etThreshold;
      unsigned int etaRange;
      unsigned int phiRange;

      unsigned int etaWindowLower;
      unsigned int etaWindowUpper;
      unsigned int etaWindowVetoLower;
      unsigned int etaWindowVetoUpper;

      unsigned int phiWindowLower;
      unsigned int phiWindowUpper;
      unsigned int phiWindowVetoLower;
      unsigned int phiWindowVetoUpper;

    };

    /// typedef for correlation parameters
    struct CorrelationParameter
    {
        unsigned long long deltaEtaRange;

        unsigned long long deltaPhiRange;
        unsigned int deltaPhiMaxbits;

      unsigned int deltaEtaRangeLower;
      unsigned int deltaEtaRangeUpper;

      unsigned int deltaPhiRangeLower;
      unsigned int deltaPhiRangeUpper;

    };


public:

    inline const std::vector<ObjectParameter>* objectParameter() const
    {
        return &m_objectParameter;
    }

    inline const CorrelationParameter* correlationParameter() const
    {
        return &m_correlationParameter;
    }


    /// set functions
    void setConditionParameter(const std::vector<ObjectParameter>& objParameter,
                               const CorrelationParameter& corrParameter);


    /// print the condition
    virtual void print(std::ostream& myCout) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const CaloTemplate&);

protected:

    /// copy function for copy constructor and operator=
    void copy( const CaloTemplate& cp);


protected:

    /// variables containing the parameters
    std::vector<ObjectParameter> m_objectParameter;
    CorrelationParameter m_correlationParameter;

};

#endif
