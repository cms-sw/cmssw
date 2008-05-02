#include "RecoEgamma/ElectronIdentification/plugins/ElectronLikelihoodESSource.h"

ElectronLikelihoodESSource::ElectronLikelihoodESSource (const edm::ParameterSet& cfg) :
  m_eleWeight (cfg.getParameter<double> ("eleWeight")) ,
  m_piWeight  (cfg.getParameter<double> ("piWeight")) ,
  m_signalWeightSplitting (cfg.getParameter<std::string> ("signalWeightSplitting")) ,
  m_backgroundWeightSplitting (cfg.getParameter<std::string> ("backgroundWeightSplitting")) ,
  m_splitSignalPdfs (cfg.getParameter<bool> ("splitSignalPdfs")) ,
  m_splitBackgroundPdfs (cfg.getParameter<bool> ("splitBackgroundPdfs"))
{
  setWhatProduced (this) ;
  findingRecord<ElectronLikelihoodRcd> () ;

  m_fisherEBLt15.push_back (cfg.getParameter<double> ("fisherCoeffEBLt15_constant")) ;
  m_fisherEBLt15.push_back (cfg.getParameter<double> ("fisherCoeffEBLt15_sigmaEtaEta")) ;
  m_fisherEBLt15.push_back (cfg.getParameter<double> ("fisherCoeffEBLt15_s9s25")) ;
  m_fisherEBLt15.push_back (cfg.getParameter<double> ("fisherCoeffEBLt15_etaLat")) ;
  m_fisherEBLt15.push_back (cfg.getParameter<double> ("fisherCoeffEBLt15_a20")) ;
  m_fisherEBGt15.push_back (cfg.getParameter<double> ("fisherCoeffEBGt15_constant")) ;
  m_fisherEBGt15.push_back (cfg.getParameter<double> ("fisherCoeffEBGt15_sigmaEtaEta")) ;
  m_fisherEBGt15.push_back (cfg.getParameter<double> ("fisherCoeffEBGt15_s9s25")) ;
  m_fisherEBGt15.push_back (cfg.getParameter<double> ("fisherCoeffEBGt15_etaLat")) ;
  m_fisherEBGt15.push_back (cfg.getParameter<double> ("fisherCoeffEBGt15_a20")) ;
  m_fisherEELt15.push_back (cfg.getParameter<double> ("fisherCoeffEELt15_constant")) ;
  m_fisherEELt15.push_back (cfg.getParameter<double> ("fisherCoeffEELt15_sigmaEtaEta")) ;
  m_fisherEELt15.push_back (cfg.getParameter<double> ("fisherCoeffEELt15_s9s25")) ;
  m_fisherEELt15.push_back (cfg.getParameter<double> ("fisherCoeffEELt15_etaLat")) ;
  m_fisherEELt15.push_back (cfg.getParameter<double> ("fisherCoeffEELt15_a20")) ;
  m_fisherEEGt15.push_back (cfg.getParameter<double> ("fisherCoeffEEGt15_constant")) ;
  m_fisherEEGt15.push_back (cfg.getParameter<double> ("fisherCoeffEEGt15_sigmaEtaEta")) ;
  m_fisherEEGt15.push_back (cfg.getParameter<double> ("fisherCoeffEEGt15_s9s25")) ;
  m_fisherEEGt15.push_back (cfg.getParameter<double> ("fisherCoeffEEGt15_etaLat")) ;
  m_fisherEEGt15.push_back (cfg.getParameter<double> ("fisherCoeffEEGt15_a20")) ;

  m_eleEBFracsLt15.push_back (cfg.getParameter<double> ("eleEBGoldenFracLt15")) ;
  m_eleEBFracsLt15.push_back (cfg.getParameter<double> ("eleEBBigbremFracLt15")) ;
  m_eleEBFracsLt15.push_back (cfg.getParameter<double> ("eleEBNarrowFracLt15")) ;
  m_eleEBFracsLt15.push_back (cfg.getParameter<double> ("eleEBShoweringFracLt15")) ;
  m_eleEBFracsGt15.push_back (cfg.getParameter<double> ("eleEBGoldenFracGt15")) ;
  m_eleEBFracsGt15.push_back (cfg.getParameter<double> ("eleEBBigbremFracGt15")) ;
  m_eleEBFracsGt15.push_back (cfg.getParameter<double> ("eleEBNarrowFracGt15")) ;
  m_eleEBFracsGt15.push_back (cfg.getParameter<double> ("eleEBShoweringFracGt15")) ;
  m_piEBFracsLt15.push_back  (cfg.getParameter<double> ("piEBGoldenFracLt15")) ;
  m_piEBFracsLt15.push_back  (cfg.getParameter<double> ("piEBBigbremFracLt15")) ;
  m_piEBFracsLt15.push_back  (cfg.getParameter<double> ("piEBNarrowFracLt15")) ;
  m_piEBFracsLt15.push_back  (cfg.getParameter<double> ("piEBShoweringFracLt15")) ;
  m_piEBFracsGt15.push_back  (cfg.getParameter<double> ("piEBGoldenFracGt15")) ;
  m_piEBFracsGt15.push_back  (cfg.getParameter<double> ("piEBBigbremFracGt15")) ;
  m_piEBFracsGt15.push_back  (cfg.getParameter<double> ("piEBNarrowFracGt15")) ;
  m_piEBFracsGt15.push_back  (cfg.getParameter<double> ("piEBShoweringFracGt15")) ;

  m_eleEEFracsLt15.push_back (cfg.getParameter<double> ("eleEEGoldenFracLt15")) ;
  m_eleEEFracsLt15.push_back (cfg.getParameter<double> ("eleEEBigbremFracLt15")) ;
  m_eleEEFracsLt15.push_back (cfg.getParameter<double> ("eleEENarrowFracLt15")) ;
  m_eleEEFracsLt15.push_back (cfg.getParameter<double> ("eleEEShoweringFracLt15")) ;
  m_piEEFracsLt15.push_back (cfg.getParameter<double> ("piEEGoldenFracLt15")) ;
  m_piEEFracsLt15.push_back (cfg.getParameter<double> ("piEEBigbremFracLt15")) ;
  m_piEEFracsLt15.push_back (cfg.getParameter<double> ("piEENarrowFracLt15")) ;
  m_piEEFracsLt15.push_back (cfg.getParameter<double> ("piEEShoweringFracLt15")) ;
  m_eleEEFracsGt15.push_back (cfg.getParameter<double> ("eleEEGoldenFracGt15")) ;
  m_eleEEFracsGt15.push_back (cfg.getParameter<double> ("eleEEBigbremFracGt15")) ;
  m_eleEEFracsGt15.push_back (cfg.getParameter<double> ("eleEENarrowFracGt15")) ;
  m_eleEEFracsGt15.push_back (cfg.getParameter<double> ("eleEEShoweringFracGt15")) ;
  m_piEEFracsGt15.push_back (cfg.getParameter<double> ("piEEGoldenFracGt15")) ;
  m_piEEFracsGt15.push_back (cfg.getParameter<double> ("piEEBigbremFracGt15")) ;
  m_piEEFracsGt15.push_back (cfg.getParameter<double> ("piEENarrowFracGt15")) ;
  m_piEEFracsGt15.push_back (cfg.getParameter<double> ("piEEShoweringFracGt15")) ;

  m_eleIDSwitches.m_useDeltaEtaCalo = cfg.getParameter<double> ("useDeltaEtaCalo") ;
  m_eleIDSwitches.m_useDeltaPhiIn   = cfg.getParameter<double> ("useDeltaPhiIn") ;
  m_eleIDSwitches.m_useHoverE       = cfg.getParameter<double> ("useHoverE") ;
  m_eleIDSwitches.m_useEoverPOut    = cfg.getParameter<double> ("useEoverPOut") ;
  m_eleIDSwitches.m_useShapeFisher  = cfg.getParameter<double> ("useShapeFisher") ;
  m_eleIDSwitches.m_useSigmaEtaEta  = cfg.getParameter<double> ("useSigmaEtaEta") ;
  m_eleIDSwitches.m_useE9overE25    = cfg.getParameter<double> ("useE9overE25") ;
}


// ----------------------------------------------------


ElectronLikelihoodESSource::~ElectronLikelihoodESSource() {
}


// ----------------------------------------------------


ElectronLikelihoodESSource::ReturnType
ElectronLikelihoodESSource::produce (const ElectronLikelihoodRcd & iRecord) 
{

  const ElectronLikelihoodCalibration *calibration = readPdfFromDB (iRecord) ;

  ReturnType LHAlgo (new ElectronLikelihood (&(*calibration), 
					     m_fisherEBLt15,     m_fisherEBGt15,
					     m_fisherEELt15,     m_fisherEEGt15,
					     m_eleEBFracsLt15,   m_piEBFracsLt15, 
					     m_eleEEFracsLt15,   m_piEEFracsLt15,
					     m_eleEBFracsGt15,   m_piEBFracsGt15, 
					     m_eleEEFracsGt15,   m_piEEFracsGt15,
					     m_eleWeight,        m_piWeight,
					     m_eleIDSwitches,
					     m_signalWeightSplitting, m_backgroundWeightSplitting,
					     m_splitSignalPdfs, m_splitBackgroundPdfs
					     ) ); 

  return LHAlgo;
}


// ----------------------------------------------------


void 
ElectronLikelihoodESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
					   const edm::IOVSyncValue&,
					   edm::ValidityInterval& oInterval ) {
  // the same PDF's is valid for any time
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime (), 
                                     edm::IOVSyncValue::endOfTime ()) ;
}


// ----------------------------------------------------

const ElectronLikelihoodCalibration*
ElectronLikelihoodESSource::readPdfFromDB( const ElectronLikelihoodRcd & iRecord ) {

  // setup the PDF's from DB
  const ElectronLikelihoodCalibration *calibration = 0;
  edm::ESHandle<ElectronLikelihoodCalibration> calibHandle;
  iRecord.getRecord<ElectronLikelihoodPdfsRcd>().get(calibHandle);
  calibration = calibHandle.product();

  return calibration;
}
