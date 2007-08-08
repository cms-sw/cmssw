// -*- C++ -*-
//-----------------------------------------------------------------------
//
// Package:    
//      EgammaAnalysis/ElectronIDESSource
// Description:
//      Class ElectronLikelihoodESSource
//      class defining the Event Setup sources, where to take:
//      PDFs, a priori probabilities, cluster shape Fisher setup
//      
// Original Authors:  Emanuele Di Marco, 
//                    Pietro Govoni,
//                    Chiara Ilaria Rovelli, 
//                    Paolo Meridiani
// Universita' di Roma "La Sapienza" and INFN Roma
// Universita' di Milano "Bicocca" and INFN Milano
//
// Created:  Fri Jun  25 11:25:36 CEST 2007
//
//-----------------------------------------------------------------------


#ifndef ElectronLikelihoodESSource_h
#define ElectronLikelihoodESSource_h

#include <memory>
#include <fstream>
#include <vector>
#include "boost/shared_ptr.hpp"
#include "EgammaAnalysis/ElectronIDAlgos/interface/ElectronLikelihood.h"
#include "EgammaAnalysis/ElectronIDAlgos/interface/LikelihoodSwitches.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CondFormats/DataRecord/interface/ElectronLikelihoodRcd.h"
#include <TDirectory.h>
#include <climits>


class ElectronLikelihoodESSource : public edm::ESProducer, public  edm::EventSetupRecordIntervalFinder {
 public:
  /// constructor from parameter set
  ElectronLikelihoodESSource( const edm::ParameterSet& );
  /// destructor
  ~ElectronLikelihoodESSource();
  /// define the return type
  typedef std::auto_ptr<ElectronLikelihood> ReturnType;
  /// return the particle table
  ReturnType produce( const ElectronLikelihoodRcd & );
  /// set validity interval
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey &,
		       const edm::IOVSyncValue &,
		       edm::ValidityInterval & );
  
 private:
  /// read pdf's from ROOT files
  void readPdfFromRootFile ();

 private:
  /// TDirectories with pdf's
  TDirectory *m_EBlt15dir;
  TDirectory *m_EElt15dir;
  TDirectory *m_EBgt15dir;
  TDirectory *m_EEgt15dir;

  /// input file names for EB pdf
  edm::FileInPath m_pdfEBlt15FileName ;
  edm::FileInPath m_pdfEBgt15FileName ;
  /// input file names for EE pdf
  edm::FileInPath m_pdfEElt15FileName ;
  edm::FileInPath m_pdfEEgt15FileName ;

  /// input file names for EB/EE Fisher coefficients
  edm::FileInPath m_fisherEBFileName ;
  edm::FileInPath m_fisherEEFileName ;

  //! barrel electron classes fractions
  std::vector<double> m_eleEBFracsLt15 ;
  std::vector<double> m_eleEBFracsGt15 ;
  //! barrel electron classes fractions for pions
  std::vector<double> m_piEBFracsLt15 ;
  std::vector<double> m_piEBFracsGt15 ;

  //! endcap electron classes fractions
  std::vector<double> m_eleEEFracsLt15 ;
  std::vector<double> m_eleEEFracsGt15 ;
  //! endcap electron classes fractions for pions
  std::vector<double> m_piEEFracsLt15 ;
  std::vector<double> m_piEEFracsGt15 ;

  //! general parameters of the id algorithm
  LikelihoodSwitches m_eleIDSwitches ;
  
  //! electrons weight
  double m_eleWeight ;
  //! pions weight
  double m_piWeight ;

};
#endif
