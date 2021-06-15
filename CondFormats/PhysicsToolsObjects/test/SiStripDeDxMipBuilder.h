#ifndef SiStripDeDxMipBuilder_H
#define SiStripDeDxMipBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
//#include "FWCore/Utilities/interface/FileInPath.h"

#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"

//#include "CLHEP/Random/RandFlat.h"
//#include "CLHEP/Random/RandGauss.h"

class SiStripDeDxMipBuilder : public edm::EDAnalyzer {
public:
  explicit SiStripDeDxMipBuilder(const edm::ParameterSet& iConfig);

  ~SiStripDeDxMipBuilder(){};

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  //edm::FileInPath fp_;
  bool printdebug_;
};

#endif
