
#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include <iostream>
#include <iomanip>
#include <assert.h>
#include <math.h>

//DEFINE STATICS
const unsigned L1GctJetEtCalibrationFunction::NUMBER_ETA_VALUES = 11;
const unsigned L1GctJetEtCalibrationFunction::N_CENTRAL_ETA_VALUES = 7;

L1GctJetEtCalibrationFunction::L1GctJetEtCalibrationFunction()
  : m_corrFunType(POWER_SERIES_CORRECTION)
{
}

L1GctJetEtCalibrationFunction::~L1GctJetEtCalibrationFunction()
{
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
  os << "=== Level-1 GCT : Jet Et Calibration Function  ===" << std::endl;
  os << std::setprecision(2);
  os << "LSB for Ht scale is " << std::fixed << fn.m_htScaleLSB << ", jet veto threshold is " << fn.m_threshold << std::endl;
  if (fn.m_corrFunType == L1GctJetEtCalibrationFunction::NO_CORRECTION) {
    os << "No jet energy corrections applied" << std::endl;
  } else { 
    switch (fn.m_corrFunType)
    {
      case L1GctJetEtCalibrationFunction::POWER_SERIES_CORRECTION:
        os << "Power series energy correction for jets is enabled" << std::endl;
        break;
      case L1GctJetEtCalibrationFunction::ORCA_STYLE_CORRECTION:
        os << "ORCA-style energy correction for jets is enabled" << std::endl;
        break;
      case L1GctJetEtCalibrationFunction::PIECEWISE_CUBIC_CORRECTION:
        os << "Piecewise 3rd-order polynomial energy correction for jets is enabled" << std::endl;
        break;
      default:
        os << "Unrecognised calibration function type" << std::endl;
        break; 
    }
    os << "Non-tau jet correction coefficients" << std::endl;
    for (unsigned i=0; i<fn.m_jetCalibFunc.size(); i++){
      os << "Eta =" << std::setw(2) << i;
      if (fn.m_jetCalibFunc.at(i).empty()) {
        os << ", no non-linear correction.";
      } else {
        os << " Coefficients = ";
        for (unsigned j=0; j<fn.m_jetCalibFunc.at(i).size();j++){
          os << fn.m_jetCalibFunc.at(i).at(j) << " "; 
        }
      }
      os << std::endl;
    }
    os << "Tau jet correction coefficients" << std::endl;
    for (unsigned i=0; i<fn.m_tauCalibFunc.size(); i++){
      os << "Eta =" << std::setw(2) << i;
      if (fn.m_tauCalibFunc.at(i).empty()) {
        os << ", no non-linear correction.";
      } else {
        os << " Coefficients = ";
        for (unsigned j=0; j<fn.m_tauCalibFunc.at(i).size();j++){
          os << fn.m_tauCalibFunc.at(i).at(j) << " "; 
        }
      }
      os << std::endl;
    }
  }
  return os;
}


double L1GctJetEtCalibrationFunction::correctedEt(const double et,
                                                  const unsigned eta,
                                                  const bool tauVeto) const
{
  if ((tauVeto && eta>=NUMBER_ETA_VALUES) || (!tauVeto && eta>=N_CENTRAL_ETA_VALUES)) {
    return 0;
  } else {
    double result=0;
    if (tauVeto) {
      assert(eta<m_jetCalibFunc.size());
      result=findCorrectedEt(et, m_jetCalibFunc.at(eta));
    } else {
      assert(eta<m_tauCalibFunc.size());
      result=findCorrectedEt(et, m_tauCalibFunc.at(eta));
    }
    if (result>m_threshold) { return result; }
    else { return 0; }
  }
}


//PRIVATE FUNCTIONS
double L1GctJetEtCalibrationFunction::findCorrectedEt(const double Et, const std::vector<double>& coeffs) const
{
  double result=0;
  switch (m_corrFunType)
  {
    case POWER_SERIES_CORRECTION:
      result = powerSeriesCorrect(Et, coeffs);
      break;
    case ORCA_STYLE_CORRECTION:
      result = orcaStyleCorrect(Et, coeffs);
      break;
    case PIECEWISE_CUBIC_CORRECTION:
      result = piecewiseCubicCorrect(Et, coeffs);
      break;
    default:
      result = Et;      
  }
  return result;
}

double L1GctJetEtCalibrationFunction::powerSeriesCorrect(const double Et, const std::vector<double>& coeffs) const
{
  double corrEt = Et;
  for (unsigned i=0; i<coeffs.size();i++) {
    corrEt += coeffs.at(i)*pow(Et,(int)i); 
  }
  return corrEt;
}

double L1GctJetEtCalibrationFunction::orcaStyleCorrect(const double Et, const std::vector<double>& coeffs) const
{
  // The coefficients are arranged in groups of four. The first in each group is a threshold value of Et.
  std::vector<double>::const_iterator next_coeff=coeffs.begin();
  while (next_coeff != coeffs.end()) {
    double threshold = *next_coeff++;
    double A = *next_coeff++;
    double B = *next_coeff++;
    double C = *next_coeff++;
    if (Et>threshold) {
      // This function is an inverse quadratic:
      //   (input Et) = A + B*(output Et) + C*(output Et)^2
      return 2*(Et-A)/(B+sqrt(B*B-4*A*C+4*Et*C));
    }
    // If we are below all specified thresholds (or the vector is empty), return output=input.
  }
  return Et;
}

double L1GctJetEtCalibrationFunction::piecewiseCubicCorrect(const double Et, const std::vector<double>& coeffs) const
{
  // The correction fuction is a set of 3rd order polynomials
  //    Et_out = Et_in + (p0 + p1*Et_in + p2*Et_in^2 + p3*Et_in^3)
  // with different coefficients for different energy ranges.
  // The parameters are arranged in groups of five.
  // The first in each group is a threshold value of input Et,
  // followed by the four coefficients for the cubic function.
  double etOut = Et;
  std::vector<double>::const_iterator next_coeff=coeffs.begin();
  while (next_coeff != coeffs.end()) {

    // Read the coefficients from the vector
    double threshold = *next_coeff++;
    double A = *next_coeff++; //p0
    double B = *next_coeff++; //p1
    double C = *next_coeff++; //p2
    double D = *next_coeff++; //p3

    // Check we are in the right energy range and make correction
    if (Et>threshold) {
      etOut += (A + etOut*(B + etOut*(C + etOut*D))) ;
      break;
    }

  }
  return etOut;
}

/// Convert the corrected Et value to an integer Et for Ht summing
uint16_t L1GctJetEtCalibrationFunction::calibratedEt(const double correctedEt) const
{
  double scaledEt = correctedEt / m_htScaleLSB;

  uint16_t jetEtOut = static_cast<uint16_t>(scaledEt);

  if(jetEtOut > L1CaloEtScale::linScaleMax) {
    return L1CaloEtScale::linScaleMax;
  } else {
    return jetEtOut;
  }
}
