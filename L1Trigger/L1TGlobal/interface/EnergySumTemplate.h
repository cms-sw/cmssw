#ifndef L1Trigger_L1TGlobal_EnergySumTemplate_h
#define L1Trigger_L1TGlobal_EnergySumTemplate_h

/**
 * \class EnergySumTemplate
 *
 *
 * Description: L1 Global Trigger energy-sum template.
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
class EnergySumTemplate : public GtCondition
{

public:

    // constructor
    EnergySumTemplate();

    // constructor
    EnergySumTemplate(const std::string&);

    // constructor
    EnergySumTemplate(const std::string&, const l1t::GtConditionType&);

    // copy constructor
    EnergySumTemplate(const EnergySumTemplate&);

    // destructor
    virtual ~EnergySumTemplate();

    // assign operator
    EnergySumTemplate& operator=(const EnergySumTemplate&);

public:

    /// typedef for a single object template
    struct ObjectParameter
    {
      unsigned int etThreshold;
      bool energyOverflow;

      unsigned int phiWindowLower;
      unsigned int phiWindowUpper;
      unsigned int phiWindowVetoLower;
      unsigned int phiWindowVetoUpper;

      // two words used only for ETM (ETM phi has 72 bins - two 64-bits words)
      // one word used for HTM
      unsigned long long phiRange0Word;
      unsigned long long phiRange1Word;
    };

public:

    inline const std::vector<ObjectParameter>* objectParameter() const {
        return &m_objectParameter;
    }

    /// set functions
    void setConditionParameter(const std::vector<ObjectParameter>&);

    /// print the condition
    virtual void print(std::ostream& myCout) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const EnergySumTemplate&);

private:

    /// copy function for copy constructor and operator=
    void copy(const EnergySumTemplate& cp);

private:

    /// variables containing the parameters
    std::vector<ObjectParameter> m_objectParameter;

};

#endif
