#ifndef SiStripDQMProfileToTkMapConverter_H
#define SiStripDQMProfileToTkMapConverter_H


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h" 
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

#include <vector>
#include <iostream>
#include <string.h>
#include <sstream>


class SiStripDQMProfileToTkMapConverter : public edm::EDAnalyzer {

public:

  SiStripDQMProfileToTkMapConverter(const edm::ParameterSet&);
  ~SiStripDQMProfileToTkMapConverter();

private:

 //Will be called at the beginning of the job
  void beginRun(const edm::Run &, const edm::EventSetup &);
  void analyze(const edm::Event&, const edm::EventSetup&){};
  void endJob();


private:
  const edm::ParameterSet conf_;
  edm::FileInPath fp_;
  SiStripDetInfoFileReader* reader;
  std::string filename, dirpath;
  std::string TkMapFileName_;

  DQMStore* dqmStore_;


  TkHistoMap *tkhisto;
  TrackerMap * tkMap;  
};
#endif
