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
//                    Chiara Ilaria Rovelli, 
//                    Paolo Meridiani
// Universita' di Roma "La Sapienza" and INFN Roma
//
// Created:  Fri Jun  25 11:25:36 CEST 2007
//
//-----------------------------------------------------------------------


#ifndef ElectronLikelihoodESSource_h
#define ElectronLikelihoodESSource_h

#include <memory>
#include <fstream>
#include <vector>
#include "RecoEgamma/ElectronIdentification/interface/ElectronLikelihood.h"
#include "RecoEgamma/ElectronIdentification/interface/LikelihoodSwitches.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CondFormats/DataRecord/interface/ElectronLikelihoodRcd.h"
#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCalibration.h"
#include <climits>
#include <string>

class ElectronLikelihoodESSource : public edm::ESProducer, public  edm::EventSetupRecordIntervalFinder {
 public:
  /// constructor from parameter set
  ElectronLikelihoodESSource( const edm::ParameterSet& );
  /// destructor
  ~ElectronLikelihoodESSource() override;
  /// define the return type
  typedef std::unique_ptr<ElectronLikelihood> ReturnType;
  /// return the particle table
  ReturnType produce( const ElectronLikelihoodRcd &);
  /// set validity interval
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey &,
		       const edm::IOVSyncValue &,
		       edm::ValidityInterval & ) override;
  
 private:
  //! read PDF's from CondDB
  const ElectronLikelihoodCalibration* readPdfFromDB ( const ElectronLikelihoodRcd & );

 private:

  //! general parameters of the id algorithm
  LikelihoodSwitches m_eleIDSwitches ;
  
  //! signal pdf splitting
  std::string m_signalWeightSplitting ;
  std::string m_backgroundWeightSplitting ;
  bool m_splitSignalPdfs ;
  bool m_splitBackgroundPdfs ;

};
#endif
