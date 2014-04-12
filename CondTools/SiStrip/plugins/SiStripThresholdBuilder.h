#ifndef CSiStripThresholdBuilder_H
#define CSiStripThresholdBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"


class SiStripThresholdBuilder : public edm::EDAnalyzer {

 public:

  explicit SiStripThresholdBuilder( const edm::ParameterSet& iConfig);

  ~SiStripThresholdBuilder(){};

  virtual void analyze(const edm::Event& , const edm::EventSetup& );

 private:
  edm::FileInPath fp_;
  uint32_t printdebug_;
};
#endif
