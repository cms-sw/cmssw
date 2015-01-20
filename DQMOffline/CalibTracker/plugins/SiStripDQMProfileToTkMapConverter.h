#ifndef SiStripDQMProfileToTkMapConverter_H
#define SiStripDQMProfileToTkMapConverter_H


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h" 
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

#include <vector>
#include <iostream>
#include <string.h>
#include <sstream>


class SiStripDQMProfileToTkMapConverter : public DQMEDHarvester {

public:

  SiStripDQMProfileToTkMapConverter(const edm::ParameterSet&);
  ~SiStripDQMProfileToTkMapConverter();

  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override;
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) {};

private:
  const edm::ParameterSet conf_;
  edm::FileInPath fp_;
  SiStripDetInfoFileReader* reader;
  std::string filename, dirpath;
  std::string TkMapFileName_;

  TkHistoMap *tkhisto;
  TrackerMap * tkMap;  
};
#endif
