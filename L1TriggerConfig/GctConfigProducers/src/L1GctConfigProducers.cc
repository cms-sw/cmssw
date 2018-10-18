#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "L1TriggerConfig/GctConfigProducers/interface/L1GctConfigProducers.h"

#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"

#include <cmath>
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
  m_tauCalibFunc(),
  m_metEtaMask(iConfig.getParameter<unsigned>("MEtEtaMask")),
  m_tetEtaMask(iConfig.getParameter<unsigned>("TEtEtaMask")),
  m_mhtEtaMask(iConfig.getParameter<unsigned>("MHtEtaMask")),
  m_thtEtaMask(iConfig.getParameter<unsigned>("HtEtaMask"))
{

   //the following lines are needed to tell the framework what
   // data is being produced
   setWhatProduced(this,&L1GctConfigProducers::produceJfParams);
   setWhatProduced(this,&L1GctConfigProducers::produceChanMask);

   //now do what ever other initialization is needed
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
   
   if (CalibStyle == "Simple") {
     m_corrFunType = 3;
     calibCoeffs = iConfig.getParameter<edm::ParameterSet>("SimpleCoefficients");
   }
   
   if (CalibStyle == "PiecewiseCubic") {
     m_corrFunType = 4;
     calibCoeffs = iConfig.getParameter<edm::ParameterSet>("PiecewiseCubicCoefficients");
   }
   
   if (CalibStyle == "PF") {
     m_corrFunType = 5;
     calibCoeffs = iConfig.getParameter<edm::ParameterSet>("PFCoefficients");
   }

   // check 
   if (CalibStyle != "None") {
     
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
     
   } else {
     // No corrections to be applied
     m_corrFunType = 0;  // no correction
     // Set the vector sizes to those expected by the CalibrationFunction
     m_jetCalibFunc.resize(L1GctJetFinderParams::NUMBER_ETA_VALUES);
     m_tauCalibFunc.resize(L1GctJetFinderParams::N_CENTRAL_ETA_VALUES);
   }

   edm::LogWarning("L1GctConfig") << "Calibration Style option " << CalibStyle << std::endl;
   
}


L1GctConfigProducers::~L1GctConfigProducers()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

// The producer methods are handled by the "Configurer" objects

L1GctConfigProducers::JfParamsReturnType 
L1GctConfigProducers::produceJfParams(const L1GctJetFinderParamsRcd& aRcd)
{
  // get geometry
  const L1CaloGeometryRecord& geomRcd = aRcd.getRecord< L1CaloGeometryRecord >() ;
  edm::ESHandle< L1CaloGeometry > geom ;
  geomRcd.get( geom ) ;
  
  // construct jet finder params object
  auto pL1GctJetFinderParams = std::make_unique<L1GctJetFinderParams>(m_rgnEtLsb,
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
								      etToEnergyConversion(geom.product()));
  
  return pL1GctJetFinderParams ;

}

L1GctConfigProducers::ChanMaskReturnType 
L1GctConfigProducers::produceChanMask(const L1GctChannelMaskRcd&) {

  L1GctChannelMask* mask = new L1GctChannelMask;

  for (unsigned ieta=0; ieta<22; ++ieta) {
    if (((m_metEtaMask>>ieta)&0x1)==1) mask->maskMissingEt(ieta);
    if (((m_tetEtaMask>>ieta)&0x1)==1) mask->maskTotalEt(ieta);
    if (((m_mhtEtaMask>>ieta)&0x1)==1) mask->maskMissingHt(ieta);
    if (((m_thtEtaMask>>ieta)&0x1)==1) mask->maskTotalHt(ieta);
  }

  return std::unique_ptr<L1GctChannelMask>(mask);

}


/// Legacy nonsense

/// Calculate Et-to-energy conversion factors for eta bins
std::vector<double> 
L1GctConfigProducers::etToEnergyConversion(
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


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1GctConfigProducers);
