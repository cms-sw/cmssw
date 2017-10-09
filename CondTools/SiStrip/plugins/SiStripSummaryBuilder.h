#ifndef SiStripSummaryBuilder_H
#define SiStripSummaryBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"


class SiStripSummaryBuilder : public edm::EDAnalyzer {

 public:

  explicit SiStripSummaryBuilder( const edm::ParameterSet& iConfig);

  ~SiStripSummaryBuilder(){};

  virtual void analyze(const edm::Event& , const edm::EventSetup& );

 private:
  edm::FileInPath fp_;
  bool printdebug_;
  
  edm::ParameterSet iConfig_;

};

#endif
