#include "L1TriggerConfig/GctConfigProducers/interface/L1GctJfParamsConfigurer.h"


#include <math.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1GctJfParamsConfigurer::L1GctJfParamsConfigurer(const edm::ParameterSet& iConfig) :
  m_rgnEtLsb(iConfig.getParameter<double>("RctRegionEtLSB")),
  m_htLsb(iConfig.getParameter<double>("GctHtLSB")),
  m_CenJetSeed(iConfig.getParameter<double>("JetFinderCentralJetSeed")),
  m_FwdJetSeed(iConfig.getParameter<double>("JetFinderForwardJetSeed")),
  m_TauJetSeed(iConfig.getParameter<double>("JetFinderCentralJetSeed")), // no separate tau jet seed yet
  m_tauIsoThresh(iConfig.getParameter<double>("TauIsoEtThreshold")),
  m_htJetThresh(iConfig.getParameter<double>("HtJetEtThreshold")),
  m_mhtJetThresh(iConfig.getParameter<double>("MHtJetEtThreshold")),
  m_EtaBoundry(7), // not programmable!
  m_corrFunType(0),
  m_convertToEnergy (iConfig.getParameter<bool>("ConvertEtValuesToEnergy")),
  m_jetCalibFunc(),
  m_tauCalibFunc()
{

   std::string CalibStyle = iConfig.getParameter<std::string>("CalibrationStyle");

  edm::ParameterSet calibCoeffs;

  if (CalibStyle == "PowerSeries") {
    m_corrFunType = 1;
    calibCoeffs = iConfig.getParameter<edm::ParameterSet>("PowerSeriesCoefficients");
  }

  if (CalibStyle == "ORCAStyle") {
    m_corrFunType = 2;
    calibCoeffs = iConfig.getParameter<edm::ParameterSet>("OrcaStyleCoefficients");
  }
  
  if (CalibStyle == "PiecewiseCubic") {
    m_corrFunType = 3;
    calibCoeffs = iConfig.getParameter<edm::ParameterSet>("PiecewiseCubicCoefficients");
  }

  if ((CalibStyle == "PowerSeries") || (CalibStyle == "ORCAStyle") || (CalibStyle == "PiecewiseCubic")) {

    // Read the coefficients from file
    // coefficients for non-tau jet corrections
    for (unsigned i=0; i<L1GctJetFinderParams::NUMBER_ETA_VALUES; ++i) {
      std::stringstream ss;
      std::string str;
      ss << "nonTauJetCalib" << i;
      ss >> str;
      m_jetCalibFunc.push_back(calibCoeffs.getParameter< std::vector<double> >(str));
    }
    // coefficients for tau jet corrections
    for (unsigned i=0; i<L1GctJetFinderParams::N_CENTRAL_ETA_VALUES; ++i) {
      std::stringstream ss;
      std::string str;
      ss << "tauJetCalib" << i;
      ss >> str;
      m_tauCalibFunc.push_back(calibCoeffs.getParameter< std::vector<double> >(str));
    }

    if (m_corrFunType==2)  // ORCA style
      { setOrcaStyleParams(); }

    if (m_corrFunType==3)  // Power series
      { setPiecewiseCubicParams(); }

  } else {
    // No corrections to be applied
    m_corrFunType = 0;  // no correction
    // Set the vector sizes to those expected by the CalibrationFunction
    m_jetCalibFunc.resize(L1GctJetFinderParams::NUMBER_ETA_VALUES);
    m_tauCalibFunc.resize(L1GctJetFinderParams::N_CENTRAL_ETA_VALUES);
    if (CalibStyle != "None") {
      edm::LogWarning("L1GctConfig") << "Unrecognised Calibration Style option " << CalibStyle
                                      << "; no Level-1 jet corrections will be applied" << std::endl;
    }
  }
                              
}


L1GctJfParamsConfigurer::~L1GctJfParamsConfigurer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
    

// ------------ methods called to produce the data  ------------
L1GctJfParamsConfigurer::JfParamsReturnType
L1GctJfParamsConfigurer::produceJfParams(const L1CaloGeometry* geom)
{
   boost::shared_ptr<L1GctJetFinderParams> pL1GctJetFinderParams =
     boost::shared_ptr<L1GctJetFinderParams> (new L1GctJetFinderParams(m_rgnEtLsb,
								       m_htLsb,
								       m_CenJetSeed,
                                                                       m_FwdJetSeed,
                                                                       m_TauJetSeed,
								       m_tauIsoThresh,
								       m_htJetThresh,
								       m_mhtJetThresh,
                                                                       m_EtaBoundry,
								       m_corrFunType,
								       m_jetCalibFunc,
								       m_tauCalibFunc,
								       m_convertToEnergy,
								       etToEnergyConversion(geom)) );

   return pL1GctJetFinderParams ;
}



//--------------------------------------------------------------------------
//
// For ORCA-style calibration, we extend the calibration function downwards
// in energy in an automated way here.
void L1GctJfParamsConfigurer::setOrcaStyleParams()
{
  for (unsigned i=0; i<m_jetCalibFunc.size(); ++i) {
    setOrcaStyleParamsForBin(m_jetCalibFunc.at(i));
  }
  for (unsigned i=0; i<m_tauCalibFunc.size(); ++i) {
    setOrcaStyleParamsForBin(m_tauCalibFunc.at(i));
  }
}

// The ORCA-style calibration function is a series of inverted quadratic functions 
// (ie x = A + B.y + C.y^2)
//
// This assumes that just one high-energy set of parameters is supplied, together with 
// an energy threshold value (in terms of y, the "true" jet Et). It calculates a set of
// parameters to be applied to lower-Et jets subject to the following constraints:
//  (i) The calibration function is continuous at the high threshold value;
//  (ii) Its slope is also continuous at the high threshold value;
//  (iii) At the low (zero-suppression) threshold, the calibration function returns y=x.
//
void L1GctJfParamsConfigurer::setOrcaStyleParamsForBin(std::vector<double>& paramsForBin)
{
  assert (paramsForBin.size() == 4);

  // The threshold for using the high-energy coefficients is
  // Measured Et=x2; Calibrated Et=y2.
  // The value of x2 is supplied from the .cfi file.
  double x2 = paramsForBin.at(0);
  double A  = paramsForBin.at(1);
  double B  = paramsForBin.at(2);
  double C  = paramsForBin.at(3);

  double y2 = 2*(x2-A)/(B + sqrt(B*B - 4*(A-x2)*C));

  // The thresold for using the low-energy coefficients is
  // Measured Et=x1; Calibrated Et=y1.
  // Here we set x1=y1=zero-suppression threshold
  double x1 = 0.;
  double y1 = x1;

  // Calculate the low-energy coefficients given (x1, y1) and y2.
  double g = (x1 - (A + y1*(B+y1*C)))/(pow((y2-y1),2));
  A = A + g*y2*y2;
  B = B - 2.0*g*y2;
  C = C + g;

  // Add the new threshold and coefficients to the end of the list.
  paramsForBin.push_back(x1);
  paramsForBin.push_back(A);
  paramsForBin.push_back(B);
  paramsForBin.push_back(C);

}

//--------------------------------------------------------------------------
//
// For ORCA-style calibration, we extend the calibration function downwards
// in energy in an automated way here.
void L1GctJfParamsConfigurer::setPiecewiseCubicParams()
{
  for (unsigned i=0; i<m_jetCalibFunc.size(); ++i) {
    setPiecewiseCubicParamsForBin(m_jetCalibFunc.at(i));
  }
  for (unsigned i=0; i<m_tauCalibFunc.size(); ++i) {
    setPiecewiseCubicParamsForBin(m_tauCalibFunc.at(i));
  }
}

// The piecewise cubic parametrisation is a series of third-order polynomials. 
// Each polynomial gives a correction to be ADDED to the raw jet Et, so that
//    Et_out = Et_in + poly(Et_in, {params} )
// Each polynomial also has an associated range of validity, specified
// in terms of a threshold value of Et_in.
//
// This assumes that the first parameter in the list is a maximum Et_in, above
// which the calibration hasn't been studied. The second is the minimum Et_in for
// the first set of polynomial coefficients. For Et values above the maximum, we
// set the function to perform a linear extrapolation. 
//
// We also perform checks here on the continuity of the parametrisation
//
void L1GctJfParamsConfigurer::setPiecewiseCubicParamsForBin(std::vector<double>& paramsForBin)
{
  unsigned numberOfPars   = paramsForBin.size();
  unsigned numberOfPieces = (numberOfPars-1)/5;
  // Check that we have a sensible number of parameters
  if ( ((numberOfPars-1) % 5) == 0) {
    std::vector<double>::const_iterator par = paramsForBin.begin();
    double etMax = *par++;

    // Check the parameters read from file.
    // Copy them into vectors for the five different parameter types.
    // The vectors are initialised with a size of 1, to hold the
    // coefficients for the extrapolation above etMax.
    std::vector<double> threshold(1), p0(1), p1(1), p2(1), p3(1);
    while (par != paramsForBin.end()) {
      threshold.push_back(*par++);
      p0.push_back(*par++);
      p1.push_back(*par++);
      p2.push_back(*par++);
      p3.push_back(*par++);
    }

    // Here's the extrapolation above etMax
    double etMaxCorr = p0.at(1) + etMax*(p1.at(1) + etMax*(p2.at(1) + etMax*p3.at(1)));
    threshold.at(0) = etMax;
    p0.at(0) = 0.0;
    p1.at(0) = etMaxCorr/etMax;
    p2.at(0) = 0.0;
    p3.at(0) = 0.0;

    // Here's the continuity check
    for (unsigned piece=1; piece<(numberOfPieces-1); piece++) {
      double et = threshold.at(piece);
      double A  = p0.at(piece) - p0.at(piece+1);
      double B  = p1.at(piece) - p1.at(piece+1);
      double C  = p2.at(piece) - p2.at(piece+1);
      double D  = p3.at(piece) - p3.at(piece+1);
      // Find the difference between the two pieces of the function
      // above and below the threshold
      double check = A + et*(B + et*(C + et*D));
      // How much discontinuity to allow? Try this ...
      if (fabs(check)>0.1) {
        edm::LogError ("L1GctConfig") << "Error reading parameters from file for piecewise cubic calibration.\n"
                                      << "The function is discontinuous at threshold no. " << piece
                                      << ", et value " << et << " GeV, by an amount " << check << " GeV"; 
      }
    }

    // Put the parameters back into the vector, with
    // the high-Et extrapolation at the beginning
    paramsForBin.clear();
    for (unsigned piece=0; piece<=numberOfPieces; piece++) {
      paramsForBin.push_back(threshold.at(piece));
      paramsForBin.push_back(p0.at(piece));
      paramsForBin.push_back(p1.at(piece));
      paramsForBin.push_back(p2.at(piece));
      paramsForBin.push_back(p3.at(piece));
    }
  } else {
    // The number of parameters is wrong
    edm::LogError ("L1GctConfig") << "Error reading parameters from file for piecewise cubic calibration.\n"
                                  << "The number of parameters is "
                                  << numberOfPars << ", but we need five parameters per piece, plus one";
  }
}

/// Calculate Et-to-energy conversion factors for eta bins
std::vector<double> L1GctJfParamsConfigurer::etToEnergyConversion(
   const L1CaloGeometry* geom) const {
  //  L1CaloGeometry* geom = new L1CaloGeometry();
  std::vector<double> result;
  // Factors for central eta bins
  for (unsigned ieta=0; ieta<7; ieta++) {
    double bineta = geom->etaBinCenter(ieta, true);
    double factor = 0.5*(exp(bineta)+exp(-bineta)); // Conversion from eta to cosec(theta)
    result.push_back(factor);
  }
  // Factors for forward eta bins
  for (unsigned ieta=0; ieta<4; ieta++) {
    double bineta = geom->etaBinCenter(ieta, false);
    double factor = 0.5*(exp(bineta)+exp(-bineta)); // Conversion from eta to cosec(theta)
    result.push_back(factor);
  }
  return result;
}
