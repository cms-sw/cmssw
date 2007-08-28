
#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"

#include <iostream>
#include <iomanip>

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


double L1GctJetEtCalibrationFunction::correctedEt(double et, unsigned eta, bool tauVeto) const {

  if ((tauVeto && eta>=NUMBER_ETA_VALUES) || (!tauVeto && eta>=N_CENTRAL_ETA_VALUES)) {
    return 0;
  } else {
    double result=0;
    // TEMP
    if (m_corrFunType==OLD_ORCA_STYLE_CORRECTION) {
      unsigned gctEta = eta + NUMBER_ETA_VALUES;
      result = static_cast<double>(orcaCalibFn( static_cast<float>(et), gctEta ));
    } else {
    if (tauVeto) {
      assert(eta<m_jetCalibFunc.size());
      result=findCorrectedEt(et, m_jetCalibFunc.at(eta));
    } else {
      assert(eta<m_tauCalibFunc.size());
      result=findCorrectedEt(et, m_tauCalibFunc.at(eta));
    }
    } //end block for TEMP
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

// This is the original calibration function
float L1GctJetEtCalibrationFunction::orcaCalibFn(float et, unsigned eta) const {

  float domainbin_L[22] = {6.52223337753073373e+00,6.64347505748981959e+00,6.78054870174118296e+00,6.75191887554567405e+00,6.60891660595437802e+00,6.57813476381055473e+00,6.96764764481347232e+00,6.77192746888150943e+00,7.16209661824076260e+00,7.09640803784948027e+00,7.29886808171882517e+00,7.29883431473330546e+00,7.24561741344293875e+00,7.05381822724987995e+00,6.52340799679028827e+00,6.96091042775473401e+00,6.69803071767842262e+00,7.79138848427964259e+00,6.78565437835616603e+00,6.71201461174192904e+00,6.60832257380386334e+00,6.54875448717649267e+00};
  
  float domainbin_U[22] = {8.90225568813317558e+00,1.24483653543590922e+01,1.32037091554958987e+01,1.70036104608977681e+01,3.54325008263432011e+01,4.28758696753095450e+01,4.73079850563588025e+01,4.74182802251108981e+01,4.62509826468679748e+01,5.02198002212212913e+01,4.69817029938948352e+01,4.77263481299233732e+01,4.86083837976362076e+01,4.80105593452927337e+01,5.11550616006504200e+01,4.90703092811585861e+01,4.11879629179572788e+01,3.21820720507165845e+01,1.71844078553560529e+01,1.33158534849654764e+01,1.43586396719878149e+01,1.08629843894704248e+01};
  
  float A[22] = {2.03682,-4.36824,-4.45258,-6.76524,-22.5984,-24.5289,-24.0313,-22.1896,-21.7818,-22.9882,-20.3321,-21.0595,-22.1007,-22.658,-23.6898,-24.8888,-23.3246,-21.5343,-6.41221,-4.58952,-3.17222,0.637666};
  
  float B[22] = {0.226303,0.683099,0.704578,0.704623,0.825928,0.80086,0.766475,0.726059,0.760964,0.792227,0.789188,0.795219,0.781097,0.768022,0.740101,0.774782,0.788106,0.814502,0.686877,0.709556,0.628581,0.317453};
  
  float C[22] = {0.00409083,0.000235995,8.22958e-05,2.47567e-05,0.000127995,0.000132914,0.000133342,0.000133035,0.000125993,8.25968e-05,9.94442e-05,9.11652e-05,0.000109351,0.000115883,0.00011112,0.000122559,0.00015868,0.000152601,0.000112927,6.29999e-05,0.000741798,0.00274605};
  
  float D[22] = {8.24022,7.55916,7.16448,6.31577,5.96339,5.31203,5.35456,4.95243,5.34809,4.93339,5.05723,5.08575,5.18643,5.15202,4.48249,5.2734,5.51785,8.00182,6.21742,6.96692,7.22975,8.12257};
  
  float E[22] = {-0.343598,-0.294067,-0.22529,-0.0718625,0.004164,0.081987,0.124964,0.15006,0.145201,0.182151,0.187525,0.184763,0.170689,0.155268,0.174603,0.133432,0.0719798,-0.0921457,-0.0525274,-0.208415,-0.253542,-0.318476};
  
  float F[22] = {0.0171799,0.0202499,0.0186897,0.0115477,0.00603883,0.00446235,0.00363449,0.00318894,0.00361997,0.00341508,0.00366392,0.0036545,0.00352303,0.00349116,0.00294891,0.00353187,0.00460384,0.00711028,0.0109351,0.0182924,0.01914,0.0161094};
  
  if(et >= domainbin_U[eta]){
    return 2*(et-A[eta])/(B[eta]+sqrt(B[eta]*B[eta]-4*A[eta]*C[eta]+4*et*C[eta]));
  }
  else if(et > domainbin_L[eta]){
    return 2*(et-D[eta])/(E[eta]+sqrt(E[eta]*E[eta]-4*D[eta]*F[eta]+4*et*F[eta]));
  }
  else return et;
  
}
//END DEBUG
