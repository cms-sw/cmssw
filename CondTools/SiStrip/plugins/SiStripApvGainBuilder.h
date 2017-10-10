#ifndef SiStripApvGainBuilder_H
#define SiStripApvGainBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"


class SiStripApvGainBuilder : public edm::EDAnalyzer {

 public:

  explicit SiStripApvGainBuilder( const edm::ParameterSet& iConfig);

  ~SiStripApvGainBuilder() override{};

  void analyze(const edm::Event& , const edm::EventSetup& ) override;

 private:
  edm::FileInPath fp_;
  bool printdebug_;
};

#endif
