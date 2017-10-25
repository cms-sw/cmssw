 #ifndef SiStripBadChannelBuilder_H
#define SiStripBadChannelBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

#include <vector>
#include <ext/hash_map>

class SiStripBadChannelBuilder : public ConditionDBWriter<SiStripBadStrip> {

public:

  explicit SiStripBadChannelBuilder(const edm::ParameterSet&);
  ~SiStripBadChannelBuilder() override;

  void algoAnalyze(const edm::Event & event, const edm::EventSetup& iSetup) override;

private:

  SiStripBadStrip* getNewObject() override{return obj;}

private:
  edm::FileInPath fp_;
  bool printdebug_;
  SiStripBadStrip* obj ;

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters BadComponentList_;
  
};
#endif
