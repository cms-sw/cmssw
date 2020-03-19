#ifndef CSiStripPedestalsBuilder_H
#define CSiStripPedestalsBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

class SiStripPedestalsBuilder : public edm::EDAnalyzer {
public:
  explicit SiStripPedestalsBuilder(const edm::ParameterSet& iConfig);

  ~SiStripPedestalsBuilder() override{};

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::FileInPath fp_;
  uint32_t printdebug_;
};
#endif
