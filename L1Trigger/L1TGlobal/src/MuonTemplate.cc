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
 *          Vladimir Rekovic - extend for indexing
 *          Rick Cavanaugh - extend for displaced muons
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

// forward declarations

// constructors
MuonTemplate::MuonTemplate() : GlobalCondition() { m_condCategory = l1t::CondMuon; }

MuonTemplate::MuonTemplate(const std::string& cName) : GlobalCondition(cName) { m_condCategory = l1t::CondMuon; }

MuonTemplate::MuonTemplate(const std::string& cName, const l1t::GtConditionType& cType)
    : GlobalCondition(cName, l1t::CondMuon, cType) {
  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectParameter.reserve(nObjects);

    m_objectType.reserve(nObjects);
    m_objectType.assign(nObjects, l1t::gtMu);
  }
}

// copy constructor
MuonTemplate::MuonTemplate(const MuonTemplate& cp) : GlobalCondition(cp.m_condName) { copy(cp); }

// destructor
MuonTemplate::~MuonTemplate() {
  // empty now
}

// assign operator
MuonTemplate& MuonTemplate::operator=(const MuonTemplate& cp) {
  copy(cp);
  return *this;
}

// setConditionParameter - set the parameters of the condition
void MuonTemplate::setConditionParameter(const std::vector<ObjectParameter>& objParameter,
                                         const CorrelationParameter& corrParameter) {
  m_objectParameter = objParameter;
  m_correlationParameter = corrParameter;
}

void MuonTemplate::print(std::ostream& myCout) const {
  myCout << "\n  MuonTemplate print..." << std::endl;

  GlobalCondition::print(myCout);

  int nObjects = nrObjects();

  for (int i = 0; i < nObjects; i++) {
    myCout << std::endl;
    myCout << "  Template for object " << i << " [ hex ]" << std::endl;
    myCout << "    ptHighThreshold   = " << std::hex << m_objectParameter[i].ptHighThreshold << std::endl;
    myCout << "    ptLowThreshold    = " << std::hex << m_objectParameter[i].ptLowThreshold << std::endl;
    myCout << "    uptHighCut        = " << std::hex << m_objectParameter[i].unconstrainedPtHigh << std::endl;
    myCout << "    uptLowCut         = " << std::hex << m_objectParameter[i].unconstrainedPtLow << std::endl;
    myCout << "    indexHigh           = " << std::hex << m_objectParameter[i].indexHigh << std::endl;
    myCout << "    indexLow            = " << std::hex << m_objectParameter[i].indexLow << std::endl;
    myCout << "    enableMip         = " << std::hex << m_objectParameter[i].enableMip << std::endl;
    myCout << "    enableIso         = " << std::hex << m_objectParameter[i].enableIso << std::endl;
    myCout << "    requestIso        = " << std::hex << m_objectParameter[i].requestIso << std::endl;
    myCout << "    charge            =" << std::dec << m_objectParameter[i].charge << std::endl;
    myCout << "    qualityLUT        = " << std::hex << m_objectParameter[i].qualityLUT << std::endl;
    myCout << "    isolationLUT      = " << std::hex << m_objectParameter[i].isolationLUT << std::endl;
    myCout << "    impactParameterLUT= " << std::hex << m_objectParameter[i].impactParameterLUT << std::endl;
    //       myCout << "    etaRange          = "
    //       << std::hex << m_objectParameter[i].etaRange << std::endl;
    //       myCout << "    phiHigh           = "
    //       << std::hex << m_objectParameter[i].phiHigh << std::endl;
    //       myCout << "    phiLow            = "
    //        << std::hex << m_objectParameter[i].phiLow << std::endl;
    myCout << "    phiWindow1Lower   =" << std::hex << m_objectParameter[i].phiWindow1Lower << std::endl;
    myCout << "    phiWindow1Upper   =" << std::hex << m_objectParameter[i].phiWindow1Upper << std::endl;
    myCout << "    phiWindow2Lower   =" << std::hex << m_objectParameter[i].phiWindow2Lower << std::endl;
    myCout << "    phiWindow2Upper   =" << std::hex << m_objectParameter[i].phiWindow2Upper << std::endl;
    myCout << "    etaWindow1Lower   =" << std::hex << m_objectParameter[i].etaWindow1Lower << std::endl;
    myCout << "    etaWindow1Upper   =" << std::hex << m_objectParameter[i].etaWindow1Upper << std::endl;
    myCout << "    etaWindow2Lower   =" << std::hex << m_objectParameter[i].etaWindow2Lower << std::endl;
    myCout << "    etaWindow2Upper   =" << std::hex << m_objectParameter[i].etaWindow2Upper << std::endl;
    myCout << "    etaWindow3Lower   =" << std::hex << m_objectParameter[i].etaWindow3Lower << std::endl;
    myCout << "    etaWindow3Upper   =" << std::hex << m_objectParameter[i].etaWindow3Upper << std::endl;
  }

  if (wsc()) {
    myCout << "  Correlation parameters "
           << "[ hex ]" << std::endl;

    myCout << "    chargeCorrelation  = " << std::hex << m_correlationParameter.chargeCorrelation << std::endl;

    myCout << "    deltaEtaRange      = " << std::hex << m_correlationParameter.deltaEtaRange << std::endl;
    myCout << "    deltaPhiRange1Word = " << std::hex << m_correlationParameter.deltaPhiRange1Word << std::endl;
    myCout << "    deltaPhiRange0Word = " << std::hex << m_correlationParameter.deltaPhiRange0Word << std::endl;
    myCout << "    deltaPhiMaxbits    = " << std::hex << m_correlationParameter.deltaPhiMaxbits << std::endl;
  } else {
    if (m_condType == l1t::Type1s) {
      myCout << "  Correlation parameters "
             << "[ hex ]" << std::endl;

      myCout << "    chargeCorrelation  = " << std::hex << m_correlationParameter.chargeCorrelation << " (charge sign) "
             << std::endl;

    } else {
      myCout << "\n  Correlation parameters "
             << "[ hex ]" << std::endl;

      myCout << "    chargeCorrelation  = " << std::hex << m_correlationParameter.chargeCorrelation << std::endl;
    }
  }

  // reset to decimal output
  myCout << std::dec << std::endl;
}

void MuonTemplate::copy(const MuonTemplate& cp) {
  m_condName = cp.condName();
  m_condCategory = cp.condCategory();
  m_condType = cp.condType();
  m_objectType = cp.objectType();
  m_condGEq = cp.condGEq();
  m_condChipNr = cp.condChipNr();
  m_condRelativeBx = cp.condRelativeBx();

  m_objectParameter = *(cp.objectParameter());
  m_correlationParameter = *(cp.correlationParameter());
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const MuonTemplate& result) {
  result.print(os);
  return os;
}
