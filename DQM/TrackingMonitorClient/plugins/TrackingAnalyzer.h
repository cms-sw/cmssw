#ifndef TrackingAnalyser_H
#define TrackingAnalyser_H

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DQMStore;
class SiStripFedCabling;
class SiStripDetCabling;
class TrackingActionExecutor;
class FEDRawDataCollection;

class TrackingAnalyser: public edm::EDAnalyzer{

public:

  /// Constructor
  TrackingAnalyser(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~TrackingAnalyser();

private:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Analyze
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);

  /// Begin Luminosity Block
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) ;

  /// End Luminosity Block
  
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup);

  /// EndRun
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Endjob
  void endJob();



private:

  void checkTrackerFEDs(edm::Event const& e);

  DQMStore* dqmStore_;

  int fileSaveFrequency_;
  int staticUpdateFrequency_;
  int globalStatusFilling_;
  int shiftReportFrequency_;

  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;

  std::string outputFilePath_;
  std::string outputFileName_;

  edm::ParameterSet tkMapPSet_;
  edm::ESHandle< SiStripFedCabling > fedCabling_;
  edm::ESHandle< SiStripDetCabling > detCabling_;
  TrackingActionExecutor* actionExecutor_;

  unsigned long long m_cacheID_;
  int nLumiSecs_;
  int nEvents_;
  bool trackerFEDsFound_;
  bool endLumiAnalysisOn_;
  std::ostringstream html_out_;

};


#endif
