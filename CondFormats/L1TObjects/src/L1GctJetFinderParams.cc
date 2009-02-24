#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

#include <math.h>

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"


const unsigned L1GctJetFinderParams::NUMBER_ETA_VALUES = 11;
const unsigned L1GctJetFinderParams::N_CENTRAL_ETA_VALUES = 7;

L1GctJetFinderParams::L1GctJetFinderParams() :
  rgnEtLsb_(0.),
  htLsb_(0.),
  cenJetEtSeed_(0.),
  forJetEtSeed_(0.),
  tauJetEtSeed_(0.),
  tauIsoEtThreshold_(0.),
  htJetEtThreshold_(0.),
  mhtJetEtThreshold_(0.),
  cenForJetEtaBoundary_(0),
  corrType_(0),
  jetCorrCoeffs_(),
  tauCorrCoeffs_(),
  convertToEnergy_(false),
  energyConversionCoeffs_()
{ }

L1GctJetFinderParams::L1GctJetFinderParams(double rgnEtLsb,
					   double htLsb,
					   double cJetSeed,
					   double fJetSeed,
					   double tJetSeed,
					   double tauIsoEtThresh,
					   double htJetEtThresh,
					   double mhtJetEtThresh,
					   unsigned etaBoundary,
					   unsigned corrType,
					   std::vector< std::vector<double> > jetCorrCoeffs,
					   std::vector< std::vector<double> > tauCorrCoeffs,
					   bool convertToEnergy,
					   std::vector<double> energyConvCoeffs) :
  rgnEtLsb_(rgnEtLsb),
  htLsb_(htLsb),
  cenJetEtSeed_(cJetSeed),
  forJetEtSeed_(fJetSeed),
  tauJetEtSeed_(tJetSeed),
  tauIsoEtThreshold_(tauIsoEtThresh),
  htJetEtThreshold_(htJetEtThresh),
  mhtJetEtThreshold_(mhtJetEtThresh),
  cenForJetEtaBoundary_(etaBoundary),
  corrType_(0),
  jetCorrCoeffs_(jetCorrCoeffs),
  tauCorrCoeffs_(tauCorrCoeffs),
  convertToEnergy_(convertToEnergy),
  energyConversionCoeffs_(energyConvCoeffs)
{ }


L1GctJetFinderParams::~L1GctJetFinderParams() {}


double L1GctJetFinderParams::correctedEtGeV(const double et, 
					 const unsigned eta, 
					 const bool tauVeto) const {

  if (eta>=NUMBER_ETA_VALUES) {
    return 0;
  } else {
    double result=0;
    if ((eta>=cenForJetEtaBoundary_) || tauVeto) {
      // Use jetCorrCoeffs for central and forward jets.
      // In forward eta bins we ignore the tau flag (as in the firmware)
      result=correctionFunction(et, jetCorrCoeffs_.at(eta));
    } else {
      // Use tauCorrCoeffs for tau jets (in central eta bins)
      result=correctionFunction(et, tauCorrCoeffs_.at(eta));
    }
    if (convertToEnergy_)  { result *= energyConversionCoeffs_.at(eta); }
    else { return 0; }
  }

}


/// Convert the corrected Et value to an integer Et for Ht summing
uint16_t L1GctJetFinderParams::correctedEtGct(const double correctedEt) const
{
  double scaledEt = correctedEt / htLsb_;

  uint16_t jetEtOut = static_cast<uint16_t>(scaledEt);
  
  // TODO : clean up these statics that are littered all over the place
  if(jetEtOut > L1CaloEtScale::linScaleMax) {
    return L1CaloEtScale::linScaleMax;
  } else {
    return jetEtOut;
  }
}



// private methods
 
double L1GctJetFinderParams::correctionFunction(const double Et, const std::vector<double>& coeffs) const
{
  double result=0;
  switch (corrType_)
  {
  case 0:  // no correction
    result = Et;
    break;
  case 1:   // power series correction
    result = powerSeriesCorrect(Et, coeffs);
    break;
  case 2:  // ORCA style correction
    result = orcaStyleCorrect(Et, coeffs);
    break;
  case 3:  // piecwise cubic correction
    result = piecewiseCubicCorrect(Et, coeffs);
    break;
  default:
    result = Et;      
  }
  return result;
}

double L1GctJetFinderParams::powerSeriesCorrect(const double Et, const std::vector<double>& coeffs) const
{
  double corrEt = Et;
  for (unsigned i=0; i<coeffs.size();i++) {
    corrEt += coeffs.at(i)*pow(Et,(int)i); 
  }
  return corrEt;
}

double L1GctJetFinderParams::orcaStyleCorrect(const double Et, const std::vector<double>& coeffs) const
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

double L1GctJetFinderParams::piecewiseCubicCorrect(const double Et, const std::vector<double>& coeffs) const
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

