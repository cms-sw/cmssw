#ifndef CondFormats_L1TObjects_L1GtJetCountsTemplate_h
#define CondFormats_L1TObjects_L1GtJetCountsTemplate_h

/**
 * \class L1GtJetCountsTemplate
 *
 *
 * Description: L1 Global Trigger "jet counts" template.
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
class L1GtJetCountsTemplate : public L1GtCondition {
public:
  // constructor
  L1GtJetCountsTemplate();

  // constructor
  L1GtJetCountsTemplate(const std::string&);

  // constructor
  L1GtJetCountsTemplate(const std::string&, const L1GtConditionType&);

  // copy constructor
  L1GtJetCountsTemplate(const L1GtJetCountsTemplate&);

  // destructor
  ~L1GtJetCountsTemplate() override;

  // assign operator
  L1GtJetCountsTemplate& operator=(const L1GtJetCountsTemplate&);

public:
  /// typedef for a single object template
  struct ObjectParameter {
    ObjectParameter() : countOverflow(false) {}
    unsigned int countIndex;
    unsigned int countThreshold;

    bool countOverflow;

    COND_SERIALIZABLE;
  };

public:
  inline const std::vector<ObjectParameter>* objectParameter() const { return &m_objectParameter; }

  /// set functions
  void setConditionParameter(const std::vector<ObjectParameter>&);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GtJetCountsTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const L1GtJetCountsTemplate& cp);

private:
  /// variables containing the parameters
  std::vector<ObjectParameter> m_objectParameter;

  COND_SERIALIZABLE;
};

#endif
