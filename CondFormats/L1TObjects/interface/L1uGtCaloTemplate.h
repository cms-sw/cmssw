#ifndef CondFormats_L1TObjects_L1uGtCaloTemplate_h
#define CondFormats_L1TObjects_L1uGtCaloTemplate_h

/**
 * \class L1uGtCaloTemplate
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
#include "CondFormats/L1TObjects/interface/L1uGtCondition.h"

// forward declarations

// class declaration
class L1uGtCaloTemplate : public L1uGtCondition
{

public:

    // constructor
    L1uGtCaloTemplate();

    // constructor
    L1uGtCaloTemplate(const std::string& );

    // constructor
    L1uGtCaloTemplate(const std::string&, const l1t::L1uGtConditionType& );

    // copy constructor
    L1uGtCaloTemplate( const L1uGtCaloTemplate& );

    // destructor
    virtual ~L1uGtCaloTemplate();

    // assign operator
    L1uGtCaloTemplate& operator= (const L1uGtCaloTemplate&);

public:

    /// typedef for a single object template
    struct ObjectParameter
    {
      unsigned int etThreshold;
      unsigned int etaRange;
      unsigned int phiRange;

      int etaRangeBegin;
      int etaRangeEnd;
      int etaRangeVetoBegin;
      int etaRangeVetoEnd;

      int phiRangeBegin;
      int phiRangeEnd;
      int phiRangeVetoBegin;
      int phiRangeVetoEnd;

    };

    /// typedef for correlation parameters
    struct CorrelationParameter
    {
        unsigned long long deltaEtaRange;

        unsigned long long deltaPhiRange;
        unsigned int deltaPhiMaxbits;
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
    friend std::ostream& operator<<(std::ostream&, const L1uGtCaloTemplate&);

protected:

    /// copy function for copy constructor and operator=
    void copy( const L1uGtCaloTemplate& cp);


protected:

    /// variables containing the parameters
    std::vector<ObjectParameter> m_objectParameter;
    CorrelationParameter m_correlationParameter;

};

#endif
