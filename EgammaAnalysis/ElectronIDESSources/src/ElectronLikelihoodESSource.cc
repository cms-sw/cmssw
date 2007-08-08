#include "FWCore/Utilities/interface/EDMException.h"

#include "EgammaAnalysis/ElectronIDESSources/interface/ElectronLikelihoodESSource.h"
#include <TFile.h>
#include <TDirectory.h>


ElectronLikelihoodESSource::ElectronLikelihoodESSource (const edm::ParameterSet& cfg) :
  m_pdfEBlt15FileName (cfg.getParameter<edm::FileInPath> ("pdfEBlt15FileName")) ,
  m_pdfEBgt15FileName (cfg.getParameter<edm::FileInPath> ("pdfEBgt15FileName")) ,
  m_pdfEElt15FileName (cfg.getParameter<edm::FileInPath> ("pdfEElt15FileName")) ,
  m_pdfEEgt15FileName (cfg.getParameter<edm::FileInPath> ("pdfEEgt15FileName")) ,
  m_fisherEBFileName (cfg.getParameter<edm::FileInPath> ("fisherEBFileName")) ,
  m_fisherEEFileName (cfg.getParameter<edm::FileInPath> ("fisherEEFileName")) ,
  m_eleWeight (cfg.getParameter<double> ("eleWeight")) ,
  m_piWeight  (cfg.getParameter<double> ("piWeight"))
{
  setWhatProduced (this) ;
  findingRecord<ElectronLikelihoodRcd> () ;

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
  m_eleIDSwitches.m_useDeltaPhiIn = cfg.getParameter<double> ("useDeltaPhiIn") ;
  m_eleIDSwitches.m_useHoverE = cfg.getParameter<double> ("useHoverE") ;
  m_eleIDSwitches.m_useEoverPOut = cfg.getParameter<double> ("useEoverPOut") ;
  m_eleIDSwitches.m_useShapeFisher = cfg.getParameter<double> ("useShapeFisher") ;

}


// ----------------------------------------------------


ElectronLikelihoodESSource::~ElectronLikelihoodESSource() {
}


// ----------------------------------------------------


ElectronLikelihoodESSource::ReturnType
ElectronLikelihoodESSource::produce (const ElectronLikelihoodRcd & iRecord) 
{

  readPdfFromRootFile () ;

  //PG FIXME maybe these can be passed in the general EleIdAlgo class ctor
  //PG in the future
  ReturnType LHAlgo (new ElectronLikelihood ( m_EBlt15dir,        m_EElt15dir,
					      m_EBgt15dir,        m_EEgt15dir,
					      m_fisherEBFileName, m_fisherEEFileName,
					      m_eleEBFracsLt15,   m_piEBFracsLt15, 
					      m_eleEEFracsLt15,   m_piEEFracsLt15,
					      m_eleEBFracsGt15,   m_piEBFracsGt15, 
					      m_eleEEFracsGt15,   m_piEEFracsGt15,
					      m_eleWeight,        m_piWeight,
					      m_eleIDSwitches ) ); 

  return LHAlgo;
}


// ----------------------------------------------------


void 
ElectronLikelihoodESSource::setIntervalFor( 
					   const edm::eventsetup::EventSetupRecordKey&,
					   const edm::IOVSyncValue&,
					   edm::ValidityInterval& oInterval ) {
  // the same PDF's is valid for any time
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime (), 
                                     edm::IOVSyncValue::endOfTime ()) ;
}


// ----------------------------------------------------


void 
ElectronLikelihoodESSource::readPdfFromRootFile() {
                                           
  m_EBlt15dir = new TFile (m_pdfEBlt15FileName.fullPath ().c_str ()) ;
  m_EElt15dir = new TFile (m_pdfEElt15FileName.fullPath ().c_str ()) ;
  m_EBgt15dir = new TFile (m_pdfEBgt15FileName.fullPath ().c_str ()) ;
  m_EEgt15dir = new TFile (m_pdfEEgt15FileName.fullPath ().c_str ()) ;

  if (m_EBlt15dir == 0)
    throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
      << m_pdfEBlt15FileName << " not opened" ; 
  if (m_EBgt15dir == 0)
    throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
      << m_pdfEBgt15FileName << " not opened" ; 
  if (m_EElt15dir == 0)
    throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
      << m_pdfEElt15FileName << " not opened" ; 
  if (m_EEgt15dir == 0)
    throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
      << m_pdfEEgt15FileName << " not opened" ; 
  return ;
}

