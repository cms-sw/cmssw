#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

#include <iostream>
#include <iomanip>
#include <math.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctStaticParameters.h"

using std::ios;

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
  corrType_(corrType),
  jetCorrCoeffs_(jetCorrCoeffs),
  tauCorrCoeffs_(tauCorrCoeffs),
  convertToEnergy_(convertToEnergy),
  energyConversionCoeffs_(energyConvCoeffs)
{ 
  // check number of eta bins
  if (jetCorrCoeffs_.size() != NUMBER_ETA_VALUES ||
      tauCorrCoeffs_.size() != N_CENTRAL_ETA_VALUES ||
      energyConversionCoeffs_.size() != NUMBER_ETA_VALUES) {

    LogDebug("L1-O2O") << "GCT jet corrections constructed with " << jetCorrCoeffs_.size() << " bins, expected " << NUMBER_ETA_VALUES << std::endl;
    LogDebug("L1-O2O") << "GCT tau corrections constructed with " << tauCorrCoeffs_.size() << " bins, expected " << N_CENTRAL_ETA_VALUES << std::endl;
    LogDebug("L1-O2O") << "GCT energy corrections constructed with " << energyConversionCoeffs_.size() << " bins, expected " << NUMBER_ETA_VALUES << std::endl;

    throw cms::Exception("InconsistentConfig") << "L1GctJetFinderParams constructed with wrong number of eta bins : " << jetCorrCoeffs_.size() << " jets, " << tauCorrCoeffs_.size() << " taus, " << energyConversionCoeffs_.size() << " energy conversion bins" << std::endl;

  }

  // check number of coefficients against expectation
  unsigned expCoeffs = 0;
  if (corrType_ == 2) expCoeffs=8;

  // only correction type 1 can have a unknown number of parameters
  if (corrType_ != 1) {
    std::vector< std::vector<double> >::const_iterator itr;      
    for (itr=jetCorrCoeffs_.begin(); itr!=jetCorrCoeffs_.end(); ++itr) {
      if (itr->size() != expCoeffs) {
	throw cms::Exception("InconsistentConfig") << "L1GctJetFinderParams constructed with " << itr->size() << " jet correction coefficients, when " << expCoeffs << " expected" << std::endl;
      }
    }
    for (itr=tauCorrCoeffs_.begin(); itr!=tauCorrCoeffs_.end(); ++itr) {
      if (itr->size() != expCoeffs) {
	throw cms::Exception("InconsistentConfig") << "L1GctJetFinderParams constructed with " << itr->size() << " tau correction coefficients, when " << expCoeffs << " expected"<< std::endl;
      }
    }
  }
  
}


L1GctJetFinderParams::~L1GctJetFinderParams() {}

//---------------------------------------------------------------------------------------------
//
// set methods
//

void L1GctJetFinderParams::setRegionEtLsb (const double rgnEtLsb)
{
  rgnEtLsb_ = rgnEtLsb;
}

void L1GctJetFinderParams::setSlidingWindowParams(const double cJetSeed,
						  const double fJetSeed,
						  const double tJetSeed,
						  const unsigned etaBoundary)
{
  cenJetEtSeed_ = cJetSeed;
  forJetEtSeed_ = fJetSeed;
  tauJetEtSeed_ = tJetSeed;
  cenForJetEtaBoundary_ = etaBoundary;
}

void L1GctJetFinderParams::setJetEtCalibrationParams(const unsigned corrType,
						     const std::vector< std::vector<double> >& jetCorrCoeffs,
						     const std::vector< std::vector<double> >& tauCorrCoeffs)
{
  corrType_ = corrType;
  jetCorrCoeffs_ = jetCorrCoeffs;
  tauCorrCoeffs_ = tauCorrCoeffs;
}

void L1GctJetFinderParams::setJetEtConvertToEnergyOn(const std::vector<double>& energyConvCoeffs)
{
  convertToEnergy_ = true;
  energyConversionCoeffs_ = energyConvCoeffs;
}

void L1GctJetFinderParams::setJetEtConvertToEnergyOff()
{
  convertToEnergy_ = false;
  energyConversionCoeffs_.clear();
}

void L1GctJetFinderParams::setHtSumParams(const double htLsb,
					  const double htJetEtThresh,
					  const double mhtJetEtThresh)
{
  htLsb_ = htLsb;
  htJetEtThreshold_ = htJetEtThresh;
  mhtJetEtThreshold_ = mhtJetEtThresh;
}

void L1GctJetFinderParams::setTauAlgorithmParams(const double tauIsoEtThresh)
{
  tauIsoEtThreshold_ = tauIsoEtThresh;
}

void L1GctJetFinderParams::setParams(const double rgnEtLsb,
				     const double htLsb,
				     const double cJetSeed,
				     const double fJetSeed,
				     const double tJetSeed,
				     const double tauIsoEtThresh,
				     const double htJetEtThresh,
				     const double mhtJetEtThresh,
				     const unsigned etaBoundary,
				     const unsigned corrType,
				     const std::vector< std::vector<double> >& jetCorrCoeffs,
				     const std::vector< std::vector<double> >& tauCorrCoeffs)
{
  setRegionEtLsb (rgnEtLsb);
  setSlidingWindowParams(cJetSeed, fJetSeed, tJetSeed, etaBoundary);
  setJetEtCalibrationParams(corrType, jetCorrCoeffs, tauCorrCoeffs);
  setHtSumParams(htLsb, htJetEtThresh, mhtJetEtThresh);
  setTauAlgorithmParams(tauIsoEtThresh);
}

//---------------------------------------------------------------------------------------------
//
// Jet Et correction methods
//

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
    return result;
  }

}


/// Convert the corrected Et value to an integer Et for Ht summing
uint16_t L1GctJetFinderParams::correctedEtGct(const double correctedEt) const
{
  double scaledEt = correctedEt / htLsb_;

  uint16_t jetEtOut = static_cast<uint16_t>(scaledEt);
  
  if(jetEtOut > L1GctStaticParameters::jetCalibratedEtMax) {
    return L1GctStaticParameters::jetCalibratedEtMax;
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



std::ostream& operator << (std::ostream& os, const L1GctJetFinderParams& fn)
{
  //  os << std::setprecision(2);

  os << "=== Level-1 GCT : Jet Finder Parameters  ===" << std::endl;
  os << "RCT region LSB               : " << std::fixed << fn.getRgnEtLsbGeV() << " GeV" << std::endl;
  os << "Central jet seed threshold   : " << std::fixed << fn.getCenJetEtSeedGeV() << " GeV" << std::endl;
  os << "Tau jet seed threshold       : " << std::fixed << fn.getTauJetEtSeedGeV() << " GeV" << std::endl;
  os << "Forward jet seed threshold   : " << std::fixed << fn.getForJetEtSeedGeV() << " GeV" << std::endl;
  os << "Tau isolation threshold      : " << std::fixed << fn.getTauIsoEtThresholdGeV() << " GeV" << std::endl;
  os << "Ht jet Et threshold          : " << std::fixed << fn.getHtJetEtThresholdGeV() << " GeV" << std::endl;
  os << "MHt jet Et threshold         : " << std::fixed << fn.getMHtJetEtThresholdGeV() << " GeV" << std::endl;
  os << "Ht LSB                       : " << std::fixed << fn.getHtLsbGeV() << " GeV" << std::endl;
  os << "Central/Forward boundary     : " << std::fixed << fn.getCenForJetEtaBoundary() << std::endl;

  os << std::endl;

  os << std::setprecision(6);
  os << ios::scientific;

  os << "=== Level-1 GCT : Jet Et Calibration Function  ===" << std::endl;
  if (fn.getCorrType() == 0) {
    os << "No jet energy corrections applied" << std::endl;
  } else { 
    switch (fn.getCorrType())
    {
      case 1:
        os << "Power series energy correction for jets is enabled" << std::endl;
        break;
      case 2:
        os << "ORCA-style energy correction for jets is enabled" << std::endl;
        break;
      default:
        os << "Unrecognised calibration function type" << std::endl;
        break; 
    }
    std::vector< std::vector<double> > jetCoeffs = fn.getJetCorrCoeffs();
    std::vector< std::vector<double> > tauCoeffs = fn.getTauCorrCoeffs();

    os << "Non-tau jet correction coefficients" << std::endl;
    for (unsigned i=0; i<jetCoeffs.size(); i++){
      os << "Eta =" << std::setw(2) << i;
      if (jetCoeffs.at(i).empty()) {
        os << ", no coefficients";
      } else {
        os << " Coefficients = ";
        for (unsigned j=0; j<jetCoeffs.at(i).size();j++){
          os << jetCoeffs.at(i).at(j) << ", "; 
        }
      }
      os << std::endl;
    }
    os << "Tau jet correction coefficients" << std::endl;
    for (unsigned i=0; i<tauCoeffs.size(); i++){
      os << "Eta =" << std::setw(2) << i;
      if (tauCoeffs.at(i).empty()) {
        os << ", no coefficients";
      } else {
        os << " Coefficients = ";
        for (unsigned j=0; j<tauCoeffs.at(i).size();j++){
          os << tauCoeffs.at(i).at(j) << ", "; 
        }
      }
      os << std::endl;
    }
  }

  os.unsetf(ios::fixed | ios::scientific);

  return os;
}


