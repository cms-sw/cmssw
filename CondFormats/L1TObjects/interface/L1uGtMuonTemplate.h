#ifndef CondFormats_L1TObjects_L1uGtMuonTemplate_h
#define CondFormats_L1TObjects_L1uGtMuonTemplate_h

/**
 * \class L1uGtMuonTemplate
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
#include <string>
#include <iosfwd>

// user include files

//   base class
#include "CondFormats/L1TObjects/interface/L1uGtCondition.h"

// forward declarations

// class declaration
class L1uGtMuonTemplate : public L1uGtCondition
{

public:

    // constructor
    L1uGtMuonTemplate();

    // constructor
    L1uGtMuonTemplate(const std::string& );

    // constructor
    L1uGtMuonTemplate(const std::string&, const l1t::L1uGtConditionType& );

    // copy constructor
    L1uGtMuonTemplate( const L1uGtMuonTemplate& );

    // destructor
    virtual ~L1uGtMuonTemplate();

    // assign operator
    L1uGtMuonTemplate& operator= (const L1uGtMuonTemplate&);

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
    friend std::ostream& operator<<(std::ostream&, const L1uGtMuonTemplate&);


private:

    /// copy function for copy constructor and operator=
    void copy( const L1uGtMuonTemplate& cp);


private:

    /// variables containing the parameters
    std::vector<ObjectParameter> m_objectParameter;
    CorrelationParameter m_correlationParameter;

};

#endif
