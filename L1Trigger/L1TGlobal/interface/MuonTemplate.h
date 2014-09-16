#ifndef L1Trigger_L1TGlobal_MuonTemplate_h
#define L1Trigger_L1TGlobal_MuonTemplate_h

/**
 * \class MuonTemplate
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
#include "L1Trigger/L1TGlobal/interface/GtCondition.h"

// forward declarations

// class declaration
class MuonTemplate : public GtCondition
{

public:

    // constructor
    MuonTemplate();

    // constructor
    MuonTemplate(const std::string& );

    // constructor
    MuonTemplate(const std::string&, const l1t::GtConditionType& );

    // copy constructor
    MuonTemplate( const MuonTemplate& );

    // destructor
    virtual ~MuonTemplate();

    // assign operator
    MuonTemplate& operator= (const MuonTemplate&);

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
    friend std::ostream& operator<<(std::ostream&, const MuonTemplate&);


private:

    /// copy function for copy constructor and operator=
    void copy( const MuonTemplate& cp);


private:

    /// variables containing the parameters
    std::vector<ObjectParameter> m_objectParameter;
    CorrelationParameter m_correlationParameter;

};

#endif
