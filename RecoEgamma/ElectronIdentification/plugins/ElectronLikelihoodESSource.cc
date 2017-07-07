#include "RecoEgamma/ElectronIdentification/plugins/ElectronLikelihoodESSource.h"

ElectronLikelihoodESSource::ElectronLikelihoodESSource (const edm::ParameterSet& cfg) :
  m_signalWeightSplitting (cfg.getParameter<std::string> ("signalWeightSplitting")) ,
  m_backgroundWeightSplitting (cfg.getParameter<std::string> ("backgroundWeightSplitting")) ,
  m_splitSignalPdfs (cfg.getParameter<bool> ("splitSignalPdfs")) ,
  m_splitBackgroundPdfs (cfg.getParameter<bool> ("splitBackgroundPdfs"))
{
  setWhatProduced (this) ;
  findingRecord<ElectronLikelihoodRcd> () ;

  m_eleIDSwitches.m_useDeltaEta     = cfg.getParameter<bool> ("useDeltaEta") ;
  m_eleIDSwitches.m_useDeltaPhi     = cfg.getParameter<bool> ("useDeltaPhi") ;
  m_eleIDSwitches.m_useHoverE       = cfg.getParameter<bool> ("useHoverE") ;
  m_eleIDSwitches.m_useEoverP       = cfg.getParameter<bool> ("useEoverP") ;
  m_eleIDSwitches.m_useSigmaEtaEta  = cfg.getParameter<bool> ("useSigmaEtaEta") ;
  m_eleIDSwitches.m_useSigmaPhiPhi  = cfg.getParameter<bool> ("useSigmaPhiPhi") ;
  m_eleIDSwitches.m_useFBrem        = cfg.getParameter<bool> ("useFBrem") ;
  m_eleIDSwitches.m_useOneOverEMinusOneOverP = cfg.getParameter<bool> ("useOneOverEMinusOneOverP") ;

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
  const ElectronLikelihoodCalibration *calibration = nullptr;
  edm::ESHandle<ElectronLikelihoodCalibration> calibHandle;
  iRecord.getRecord<ElectronLikelihoodPdfsRcd>().get(calibHandle);
  calibration = calibHandle.product();

  return calibration;
}
