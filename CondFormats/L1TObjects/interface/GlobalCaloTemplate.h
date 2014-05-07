#ifndef CondFormats_L1TObjects_GlobalCaloTemplate_h
#define CondFormats_L1TObjects_GlobalCaloTemplate_h

/**
 * \class GlobalCaloTemplate
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
#include "CondFormats/L1TObjects/interface/GlobalCondition.h"

// forward declarations

// class declaration
class GlobalCaloTemplate : public GlobalCondition
{

public:

    // constructor
    GlobalCaloTemplate();

    // constructor
    GlobalCaloTemplate(const std::string& );

    // constructor
    GlobalCaloTemplate(const std::string&, const l1t::GlobalConditionType& );

    // copy constructor
    GlobalCaloTemplate( const GlobalCaloTemplate& );

    // destructor
    virtual ~GlobalCaloTemplate();

    // assign operator
    GlobalCaloTemplate& operator= (const GlobalCaloTemplate&);

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
    friend std::ostream& operator<<(std::ostream&, const GlobalCaloTemplate&);

protected:

    /// copy function for copy constructor and operator=
    void copy( const GlobalCaloTemplate& cp);


protected:

    /// variables containing the parameters
    std::vector<ObjectParameter> m_objectParameter;
    CorrelationParameter m_correlationParameter;

};

#endif
