#ifndef SiStripQualityHotStripIdentifier_H
#define SiStripQualityHotStripIdentifier_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <vector>

#include <ext/hash_map>

class SiStripQualityHotStripIdentifier : public ConditionDBWriter<SiStripBadStrip> {

public:

  explicit SiStripQualityHotStripIdentifier(const edm::ParameterSet&);
  ~SiStripQualityHotStripIdentifier();

private:

 //Will be called at the beginning of the job
  void algoBeginJob(const edm::EventSetup&){};
  //Will be called at the beginning of each run in the job
  void algoBeginRun(const edm::Run &, const edm::EventSetup &){};
  //Will be called at the beginning of each luminosity block in the run
  void algoBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){};
  //Will be called at every event
  void algoAnalyze(const edm::Event&, const edm::EventSetup&){};
  //Will be called at the end of each run in the job
  void algoEndRun(const edm::Run &, const edm::EventSetup &){};
  //Will be called at the end of the job
  void algoEndJob(){};

  SiStripBadStrip* getNewObject();


  void bookHisto(){};
  void resetHisto(){};
  

private:
  edm::FileInPath fp_;
  bool printdebug_;
  std::vector<uint32_t> BadModuleList_;
  SiStripDetInfoFileReader* reader;

};
#endif
