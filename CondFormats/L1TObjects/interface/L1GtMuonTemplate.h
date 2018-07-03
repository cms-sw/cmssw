#ifndef CondFormats_L1TObjects_L1GtMuonTemplate_h
#define CondFormats_L1TObjects_L1GtMuonTemplate_h

/**
 * \class L1GtMuonTemplate
 *
 *
 * Description: L1 Global Trigger muon template.
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
#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <iosfwd>

// user include files

//   base class
#include "CondFormats/L1TObjects/interface/L1GtCondition.h"

// forward declarations

// class declaration
class L1GtMuonTemplate : public L1GtCondition
{

public:

    // constructor
    L1GtMuonTemplate();

    // constructor
    L1GtMuonTemplate(const std::string& );

    // constructor
    L1GtMuonTemplate(const std::string&, const L1GtConditionType& );

    // copy constructor
    L1GtMuonTemplate( const L1GtMuonTemplate& );

    // destructor
    ~L1GtMuonTemplate() override;

    // assign operator
    L1GtMuonTemplate& operator= (const L1GtMuonTemplate&);

public:

    // typedef for a single object template
    struct ObjectParameter
    {
        unsigned int ptHighThreshold;
        unsigned int ptLowThreshold;
        bool enableMip;
        bool enableIso;
        bool requestIso;
        unsigned int qualityRange;
        unsigned long long etaRange;
        unsigned int phiHigh;
        unsigned int phiLow;
    
    COND_SERIALIZABLE;
};

    // typedef for correlation parameters
    // chargeCorrelation is defined always
    // see documentation for meaning
    struct CorrelationParameter
    {
        unsigned int chargeCorrelation;
        unsigned long long deltaEtaRange;

        unsigned long long deltaPhiRange0Word;
        unsigned long long deltaPhiRange1Word;
        unsigned int deltaPhiMaxbits;
    
    COND_SERIALIZABLE;
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
    void print(std::ostream& myCout) const override;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1GtMuonTemplate&);


private:

    /// copy function for copy constructor and operator=
    void copy( const L1GtMuonTemplate& cp);


private:

    /// variables containing the parameters
    std::vector<ObjectParameter> m_objectParameter;
    CorrelationParameter m_correlationParameter;


    COND_SERIALIZABLE;
};

#endif
