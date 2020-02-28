#ifndef CondFormats_L1TObjects_L1GtCastorTemplate_h
#define CondFormats_L1TObjects_L1GtCastorTemplate_h

/**
 * \class L1GtCastorTemplate
 *
 *
 * Description: L1 Global Trigger CASTOR template.
 *
 * Implementation:
 *    Instantiated L1GtCondition. CASTOR conditions sends a logical result only.
 *    No changes are possible at the L1 GT level. CASTOR conditions can be used
 *    in physics algorithms in combination with muon, calorimeter, energy sum
 *    and jet-counts conditions.
 *    It has zero objects.
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
class L1GtCastorTemplate : public L1GtCondition {
public:
  // constructor
  L1GtCastorTemplate();

  // constructor
  L1GtCastorTemplate(const std::string&);

  // constructor
  L1GtCastorTemplate(const std::string&, const L1GtConditionType&);

  // copy constructor
  L1GtCastorTemplate(const L1GtCastorTemplate&);

  // destructor
  ~L1GtCastorTemplate() override;

  // assign operator
  L1GtCastorTemplate& operator=(const L1GtCastorTemplate&);

public:
  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GtCastorTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const L1GtCastorTemplate& cp);

  COND_SERIALIZABLE;
};

#endif
