#ifndef ElectronLikelihoodESSource_h
#define ElectronLikelihoodESSource_h

#include <memory>
#include <fstream>
#include <vector>
#include "boost/shared_ptr.hpp"
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
  ~ElectronLikelihoodESSource();
  /// define the return type
  typedef std::auto_ptr<ElectronLikelihood> ReturnType;
  /// return the particle table
  ReturnType produce( const ElectronLikelihoodRcd &);
  /// set validity interval
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey &,
		       const edm::IOVSyncValue &,
		       edm::ValidityInterval & );
  
 private:
  //! read PDF's from CondDB
  const ElectronLikelihoodCalibration* readPdfFromDB ( const ElectronLikelihoodRcd & );

 private:

  //! fisher coefficients
  std::vector<double> m_fisherEBLt15;
  std::vector<double> m_fisherEBGt15;
  std::vector<double> m_fisherEELt15;
  std::vector<double> m_fisherEEGt15;

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

  //! signal pdf splitting
  std::string m_signalWeightSplitting ;
  std::string m_backgroundWeightSplitting ;
  bool m_splitSignalPdfs ;
  bool m_splitBackgroundPdfs ;

};
#endif
