
#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include <iostream>

#include <math.h>

//DEFINE STATICS
const unsigned L1GctJetEtCalibrationFunction::JET_ENERGY_BITWIDTH = 10;
const unsigned L1GctJetEtCalibrationFunction::NUMBER_ETA_VALUES = 11;
const unsigned L1GctJetEtCalibrationFunction::N_CENTRAL_ETA_VALUES = 7;

L1GctJetEtCalibrationFunction::L1GctJetEtCalibrationFunction()
{
}

L1GctJetEtCalibrationFunction::~L1GctJetEtCalibrationFunction()
{
}

void L1GctJetEtCalibrationFunction::setOutputEtScale(const L1CaloEtScale& scale) {
  m_outputEtScale = scale;
}

void L1GctJetEtCalibrationFunction::setParams(const double& htScale,
                                              const double& threshold,
                                              const std::vector< std::vector<double> >& jetCalibFunc,
                                              const std::vector< std::vector<double> >& tauCalibFunc ) {
  assert ((jetCalibFunc.size() == NUMBER_ETA_VALUES) && (tauCalibFunc.size() == N_CENTRAL_ETA_VALUES));
  m_htScaleLSB = htScale;
  m_threshold  = threshold;
  m_jetCalibFunc = jetCalibFunc;
  m_tauCalibFunc = tauCalibFunc;
}

std::ostream& operator << (std::ostream& os, const L1GctJetEtCalibrationFunction& fn)
{
  os << "===L1GctJetEtCalibrationFunction===" << std::endl;
  os << "LSB for Ht scale is " << fn.m_htScaleLSB << ", jet veto threshold is " << fn.m_threshold << std::endl;
  os << "Non-tau jet correction coefficients" << std::endl;
  for (unsigned i=0; i<fn.m_jetCalibFunc.size(); i++){
    os << "Eta = " << i << " Coefficients = ";
    for (unsigned j=0; j<fn.m_jetCalibFunc.at(i).size();j++){
      os << fn.m_jetCalibFunc.at(i).at(j) << " "; 
    }
    os << std::endl;
  }
  os << "Tau jet correction coefficients" << std::endl;
  for (unsigned i=0; i<fn.m_tauCalibFunc.size(); i++){
    os << "Eta = " << i << " Coefficients = ";
    for (unsigned j=0; j<fn.m_tauCalibFunc.at(i).size();j++){
      os << fn.m_tauCalibFunc.at(i).at(j) << " "; 
    }
    os << std::endl;
  }
  return os;
}

/// Here's the public interface to the lut values
uint16_t L1GctJetEtCalibrationFunction::lutValue(const uint16_t lutAddress) const {
  static const unsigned nEtaBits = 4;
  static const uint16_t maxEtMask  = static_cast<uint16_t>((1 << JET_ENERGY_BITWIDTH) - 1);
  static const uint16_t etaMask    = static_cast<uint16_t>((1 << nEtaBits) - 1);
  static const uint16_t tauBitMask = static_cast<uint16_t>( 1 << (JET_ENERGY_BITWIDTH + nEtaBits));
  uint16_t jetEt = lutAddress & maxEtMask;
  unsigned eta = static_cast<unsigned>((lutAddress >> JET_ENERGY_BITWIDTH) & etaMask);
  bool tauVeto = ((lutAddress & tauBitMask)==0);

  if ((tauVeto && eta>=NUMBER_ETA_VALUES) || (!tauVeto && eta>=N_CENTRAL_ETA_VALUES)) {
    return (uint16_t)0;
  } else {
    double corrEt = correctedEt(jetEt, eta, tauVeto);
    if (corrEt < m_threshold) {
      return (uint16_t)0;
    } else {
      return calibratedEt(corrEt) | (rank(corrEt) << JET_ENERGY_BITWIDTH);
    }
  }
}

//PRIVATE FUNCTIONS
/// Find the corrected Et value for this jet
double L1GctJetEtCalibrationFunction::correctedEt(const uint16_t jetEt, const unsigned eta, const bool tauVeto) const
{
  assert(m_outputEtScale.linearLsb() > 0.0);

  // Initialise the return value to the input, so that
  // an empty vector of correction factors gives a linear response
  double uncoEt = static_cast<double>(jetEt) * m_outputEtScale.linearLsb();
  if (tauVeto) {
    assert(eta<m_jetCalibFunc.size());
    return powerSeriesCorrect(uncoEt, m_jetCalibFunc.at(eta));
  } else {
    assert(eta<m_tauCalibFunc.size());
    return powerSeriesCorrect(uncoEt, m_tauCalibFunc.at(eta));
  }
}

double L1GctJetEtCalibrationFunction::powerSeriesCorrect(const double Et, const std::vector<double> coeffs) const
{
  double corrEt = Et;
  for (unsigned i=0; i<coeffs.size();i++) {
    corrEt += coeffs.at(i)*pow(Et,(int)i); 
  }
  return corrEt;
}

/// Convert the corrected Et value to a non-linear jet rank for sorting
uint16_t L1GctJetEtCalibrationFunction::rank(const double Et) const
{
  uint16_t jetRankOut = m_outputEtScale.rank(Et);

  if(jetRankOut > L1CaloEtScale::rankScaleMax) {
    return L1CaloEtScale::rankScaleMax;
  } else {
    return jetRankOut;
  }
}

/// Convert the corrected Et value to a linear Et for Ht summing
uint16_t L1GctJetEtCalibrationFunction::calibratedEt(const double Et) const
{
  double scaledEt = Et / m_htScaleLSB;

  uint16_t jetEtOut = static_cast<uint16_t>(scaledEt);

  if(jetEtOut > L1CaloEtScale::linScaleMax) {
    return L1CaloEtScale::linScaleMax;
  } else {
    return jetEtOut;
  }
}
