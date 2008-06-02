#include "FWCore/Utilities/interface/Exception.h"

#include "L1TriggerConfig/GctConfigProducers/interface/L1GctConfigProducers.h"

#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"

#include <string>
#include <math.h>

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1GctConfigProducers::L1GctConfigProducers(const edm::ParameterSet& iConfig) :
  m_CenJetSeed(iConfig.getParameter<unsigned>("JetFinderCentralJetSeed")),
  m_FwdJetSeed(iConfig.getParameter<unsigned>("JetFinderForwardJetSeed")),
  m_TauJetSeed(iConfig.getParameter<unsigned>("JetFinderCentralJetSeed")), // no separate tau jet seed yet
  m_EtaBoundry(7), // not programmable!
  m_htScaleLSB(iConfig.getParameter<double>("L1CaloHtScaleLsbInGeV")), // get the CalibrationFunction parameters
  m_threshold (iConfig.getParameter<double>("L1CaloJetZeroSuppressionThresholdInGeV")),
  m_jetCalibFunc(), m_tauCalibFunc(),
  m_corrFunType(L1GctJetEtCalibrationFunction::POWER_SERIES_CORRECTION)
{
   //the following lines are needed to tell the framework what
   // data is being produced
   setWhatProduced(this,&L1GctConfigProducers::produceCalibFun);
   setWhatProduced(this,&L1GctConfigProducers::produceJfParams);

   //now do what ever other initialization is needed

  // Options for different styles of calbration function from the config file
  std::string CalibStyle = iConfig.getParameter<std::string>("CalibrationStyle");

  edm::ParameterSet calibCoeffs;

  if (CalibStyle == "PowerSeries") {
    m_corrFunType = L1GctJetEtCalibrationFunction::POWER_SERIES_CORRECTION;
    calibCoeffs = iConfig.getParameter<edm::ParameterSet>("PowerSeriesCoefficients");
  }

  if (CalibStyle == "ORCAStyle") {
    m_corrFunType = L1GctJetEtCalibrationFunction::ORCA_STYLE_CORRECTION;
    calibCoeffs = iConfig.getParameter<edm::ParameterSet>("OrcaStyleCoefficients");
  }

  if ((CalibStyle != "PowerSeries") && (CalibStyle != "ORCAStyle")) { 
    throw cms::Exception("L1GctConfigError")
      << "L1GctConfigProducers cannot continue.\n"
      << "Invalid option " << CalibStyle << " read from configuration file.\n"
      << "Should be PowerSeries or ORCAStyle.\n";
  }

   // coefficients for non-tau jet corrections
   for (unsigned i=0; i<L1GctJetEtCalibrationFunction::NUMBER_ETA_VALUES; ++i) {
     std::stringstream ss;
     std::string str;
     ss << "nonTauJetCalib" << i;
     ss >> str;
     m_jetCalibFunc.push_back(calibCoeffs.getParameter< std::vector<double> >(str));
   }
   // coefficients for tau jet corrections
   for (unsigned i=0; i<L1GctJetEtCalibrationFunction::N_CENTRAL_ETA_VALUES; ++i) {
     std::stringstream ss;
     std::string str;
     ss << "tauJetCalib" << i;
     ss >> str;
     m_tauCalibFunc.push_back(calibCoeffs.getParameter< std::vector<double> >(str));
   }

   if (m_corrFunType==L1GctJetEtCalibrationFunction::ORCA_STYLE_CORRECTION)
     { setOrcaStyleParams(); }
}


L1GctConfigProducers::~L1GctConfigProducers()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
    

// ------------ method called to produce the data  ------------
L1GctConfigProducers::CalibFunReturnType
L1GctConfigProducers::produceCalibFun(const L1GctJetCalibFunRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1GctJetEtCalibrationFunction> pL1GctJetEtCalibrationFunction =
     boost::shared_ptr<L1GctJetEtCalibrationFunction> (new L1GctJetEtCalibrationFunction());

   pL1GctJetEtCalibrationFunction->setParams(m_htScaleLSB, m_threshold,
					     m_jetCalibFunc,
					     m_tauCalibFunc);

   pL1GctJetEtCalibrationFunction->setCorrectionFunctionType(m_corrFunType);

   return pL1GctJetEtCalibrationFunction ;
}

L1GctConfigProducers::JfParamsReturnType
L1GctConfigProducers::produceJfParams(const L1GctJetFinderParamsRcd&)
{
   using namespace edm::es;
   boost::shared_ptr<L1GctJetFinderParams> pL1GctJetFinderParams =
     boost::shared_ptr<L1GctJetFinderParams> (new L1GctJetFinderParams(m_CenJetSeed,
                                                                       m_FwdJetSeed,
                                                                       m_TauJetSeed,
                                                                       m_EtaBoundry));

   return pL1GctJetFinderParams ;
}

//--------------------------------------------------------------------------
//
// For ORCA-style calibration, we extend the calibration function downwards
// in energy in an automated way here.
void L1GctConfigProducers::setOrcaStyleParams()
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
void L1GctConfigProducers::setOrcaStyleParamsForBin(std::vector<double>& paramsForBin)
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
  double x1 = m_threshold;
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
//
//--------------------------------------------------------------------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1GctConfigProducers);
