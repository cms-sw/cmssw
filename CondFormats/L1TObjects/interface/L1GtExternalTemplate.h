#ifndef CondFormats_L1TObjects_L1GtExternalTemplate_h
#define CondFormats_L1TObjects_L1GtExternalTemplate_h

/**
 * \class L1GtExternalTemplate
 *
 *
 * Description: L1 Global Trigger external template.
 *
 * Implementation:
 *    Instantiated L1GtCondition. External conditions sends a logical result only.
 *    No changes are possible at the L1 GT level. External conditions can be used
 *    in physics algorithms in combination with other defined conditions,
 *    see L1GtFwd.
 *
 *    It has zero objects associated.
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
class L1GtExternalTemplate : public L1GtCondition
{

public:

    // constructor
    L1GtExternalTemplate();

    // constructor
    L1GtExternalTemplate(const std::string&);

    // constructor
    L1GtExternalTemplate(const std::string&, const L1GtConditionType&);

    // copy constructor
    L1GtExternalTemplate(const L1GtExternalTemplate&);

    // destructor
    ~L1GtExternalTemplate() override;

    // assign operator
    L1GtExternalTemplate& operator=(const L1GtExternalTemplate&);

public:

    /// print the condition
    void print(std::ostream& myCout) const override;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1GtExternalTemplate&);


private:

    /// copy function for copy constructor and operator=
    void copy(const L1GtExternalTemplate& cp);


    COND_SERIALIZABLE;
};

#endif
