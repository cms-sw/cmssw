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
 * \new features: Vladimir Rekovic
 *                - extend for indexing
 * \new features: Bernhard Arnold, Elisa Fontanesi                                                   
 *                - added etaWindows for the checkRangeEta function: it allows to use up to five eta cuts in L1 algorithms 
 *                - extended for muon track finder index feature (used for Run 3 muon monitoring seeds)                   
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
class MuonTemplate : public GlobalCondition {
public:
  // constructor
  MuonTemplate();

  // constructor
  MuonTemplate(const std::string&);

  // constructor
  MuonTemplate(const std::string&, const l1t::GtConditionType&);

  // copy constructor
  MuonTemplate(const MuonTemplate&);

  // destructor
  ~MuonTemplate() override;

  // assign operator
  MuonTemplate& operator=(const MuonTemplate&);

public:
  struct Window {
    unsigned int lower;
    unsigned int upper;
  };

  // typedef for a single object template
  struct ObjectParameter {
    unsigned int unconstrainedPtHigh;
    unsigned int unconstrainedPtLow;
    unsigned int impactParameterHigh;
    unsigned int impactParameterLow;
    unsigned int ptHighThreshold;
    unsigned int ptLowThreshold;
    unsigned int indexHigh;
    unsigned int indexLow;
    bool enableMip;
    bool enableIso;
    bool requestIso;
    unsigned int qualityLUT;
    unsigned int isolationLUT;
    unsigned int impactParameterLUT;
    unsigned long long etaRange;
    unsigned int phiHigh;
    unsigned int phiLow;

    int charge;

    std::vector<Window> etaWindows;

    unsigned int phiWindow1Lower;
    unsigned int phiWindow1Upper;
    unsigned int phiWindow2Lower;
    unsigned int phiWindow2Upper;

    std::vector<Window> tfMuonIndexWindows;
  };

  // typedef for correlation parameters
  // chargeCorrelation is defined always
  // see documentation for meaning
  struct CorrelationParameter {
    unsigned int chargeCorrelation;
    //unsigned long long deltaEtaRange;

    unsigned long long deltaPhiRange0Word;
    unsigned long long deltaPhiRange1Word;
    //unsigned int deltaPhiMaxbits;

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
  friend std::ostream& operator<<(std::ostream&, const MuonTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const MuonTemplate& cp);

private:
  /// variables containing the parameters
  std::vector<ObjectParameter> m_objectParameter;
  CorrelationParameter m_correlationParameter;
};

#endif
