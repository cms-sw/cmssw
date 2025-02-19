#ifndef SiStripBadFiberBuilder_H
#define SiStripBadFiberBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <vector>

#include <ext/hash_map>

class SiStripBadFiberBuilder : public ConditionDBWriter<SiStripBadStrip> {

public:

  explicit SiStripBadFiberBuilder(const edm::ParameterSet&);
  ~SiStripBadFiberBuilder();

  void algoAnalyze(const edm::Event & event, const edm::EventSetup& iSetup);

private:

  SiStripBadStrip* getNewObject(){return obj;}

private:
  edm::FileInPath fp_;
  bool printdebug_;
  SiStripBadStrip* obj ;

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters BadComponentList_;

};
#endif
