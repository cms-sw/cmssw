#ifndef L1Trigger_L1TGlobal_CaloTemplate_h
#define L1Trigger_L1TGlobal_CaloTemplate_h

/**
 * \class CaloTemplate
 *
 *
 * Description: L1 Global Trigger calo template.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *          Vladimir Rekovic - exted for indexing

 * \new features: R. Cavanaugh
 *          - added LUT bit for LLP displaced jets
 *            Note: Calo Trig considers the DISP bit part of the 
 *                  quality word, but uGT firmware considers the 
 *                  DISP bit to be distinct from the quality word.
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
#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"

// forward declarations

// class declaration
class CaloTemplate : public GlobalCondition {
public:
  // constructor
  CaloTemplate();

  // constructor
  CaloTemplate(const std::string&);

  // constructor
  CaloTemplate(const std::string&, const l1t::GtConditionType&);

  // copy constructor
  CaloTemplate(const CaloTemplate&);

  // destructor
  ~CaloTemplate() override;

  // assign operator
  CaloTemplate& operator=(const CaloTemplate&);

public:
  /// typedef for a single object template
  struct ObjectParameter {
    unsigned int etLowThreshold;
    unsigned int etHighThreshold;
    unsigned int indexLow;
    unsigned int indexHigh;
    unsigned int etaRange;
    unsigned int phiRange;

    unsigned int isolationLUT;
    unsigned int qualityLUT;
    unsigned int displacedLUT;  // Added for LLP Jets

    unsigned int etaWindow1Lower;
    unsigned int etaWindow1Upper;
    unsigned int etaWindow2Lower;
    unsigned int etaWindow2Upper;
    unsigned int etaWindow3Lower;
    unsigned int etaWindow3Upper;

    unsigned int phiWindow1Lower;
    unsigned int phiWindow1Upper;
    unsigned int phiWindow2Lower;
    unsigned int phiWindow2Upper;
  };

  /// typedef for correlation parameters
  struct CorrelationParameter {
    unsigned long long deltaEtaRange;

    unsigned long long deltaPhiRange;
    unsigned int deltaPhiMaxbits;

    unsigned int deltaEtaRangeLower;
    unsigned int deltaEtaRangeUpper;

    unsigned int deltaPhiRangeLower;
    unsigned int deltaPhiRangeUpper;
  };

public:
  inline const std::vector<ObjectParameter>* objectParameter() const { return &m_objectParameter; }

  inline const CorrelationParameter* correlationParameter() const { return &m_correlationParameter; }

  /// set functions
  void setConditionParameter(const std::vector<ObjectParameter>& objParameter,
                             const CorrelationParameter& corrParameter);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const CaloTemplate&);

protected:
  /// copy function for copy constructor and operator=
  void copy(const CaloTemplate& cp);

protected:
  /// variables containing the parameters
  std::vector<ObjectParameter> m_objectParameter;
  CorrelationParameter m_correlationParameter;
};

#endif
