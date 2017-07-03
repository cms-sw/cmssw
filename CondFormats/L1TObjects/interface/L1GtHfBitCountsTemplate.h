#ifndef CondFormats_L1TObjects_L1GtHfBitCountsTemplate_h
#define CondFormats_L1TObjects_L1GtHfBitCountsTemplate_h

/**
 * \class L1GtHfBitCountsTemplate
 *
 *
 * Description: L1 Global Trigger "HF bit counts" template.
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
class L1GtHfBitCountsTemplate : public L1GtCondition
{

public:

    // constructor
    L1GtHfBitCountsTemplate();

    // constructor
    L1GtHfBitCountsTemplate(const std::string& );

    // constructor
    L1GtHfBitCountsTemplate(const std::string&, const L1GtConditionType& );

    // copy constructor
    L1GtHfBitCountsTemplate( const L1GtHfBitCountsTemplate& );

    // destructor
    ~L1GtHfBitCountsTemplate() override;

    // assign operator
    L1GtHfBitCountsTemplate& operator= (const L1GtHfBitCountsTemplate&);

public:

    /// typedef for a single object template
    struct ObjectParameter
    {
        unsigned int countIndex;
        unsigned int countThreshold;

    
    COND_SERIALIZABLE;
};


public:

    inline const std::vector<ObjectParameter>* objectParameter() const
    {
        return &m_objectParameter;
    }


    /// set functions
    void setConditionParameter(const std::vector<ObjectParameter>&);


    /// print the condition
    void print(std::ostream& myCout) const override;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1GtHfBitCountsTemplate&);

private:

    /// copy function for copy constructor and operator=
    void copy( const L1GtHfBitCountsTemplate& cp);


private:

    /// variables containing the parameters
    std::vector<ObjectParameter> m_objectParameter;


    COND_SERIALIZABLE;
};

#endif
