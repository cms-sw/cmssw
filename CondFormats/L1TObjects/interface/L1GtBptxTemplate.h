#ifndef CondFormats_L1TObjects_L1GtBptxTemplate_h
#define CondFormats_L1TObjects_L1GtBptxTemplate_h

/**
 * \class L1GtBptxTemplate
 *
 *
 * Description: L1 Global Trigger BPTX template.
 *
 * Implementation:
 *    Instantiated L1GtCondition. BPTX conditions sends a logical result only.
 *    No changes are possible at the L1 GT level. BPTX conditions can be used
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
class L1GtBptxTemplate : public L1GtCondition
{

public:

    // constructor
    L1GtBptxTemplate();

    // constructor
    L1GtBptxTemplate(const std::string&);

    // constructor
    L1GtBptxTemplate(const std::string&, const L1GtConditionType&);

    // copy constructor
    L1GtBptxTemplate(const L1GtBptxTemplate&);

    // destructor
    ~L1GtBptxTemplate() override;

    // assign operator
    L1GtBptxTemplate& operator=(const L1GtBptxTemplate&);

public:

    /// print the condition
    void print(std::ostream& myCout) const override;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1GtBptxTemplate&);


private:

    /// copy function for copy constructor and operator=
    void copy(const L1GtBptxTemplate& cp);


    COND_SERIALIZABLE;
};

#endif
