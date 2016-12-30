#ifndef SiStripAnalyser_H
#define SiStripAnalyser_H

/** \class SiStripAnalyser
 * *
 *  SiStrip SiStripAnalyser
 *  \author  S. Dutta INFN-Pisa
 *   
 */

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
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

class SiStripWebInterface;
class SiStripFedCabling;
class SiStripDetCabling;
class SiStripActionExecutor;
class SiStripClassToMonitorCondData;
class FEDRawDataCollection;
class SiStripAnalyser: public DQMEDHarvester {

public:

  /// Constructor
  SiStripAnalyser(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripAnalyser();

  //void defaultWebPage(xgi::Input *in, xgi::Output *out); 
  //void publish(xdata::InfoSpace *){};
  //  void handleWebRequest(xgi::Input *in, xgi::Output *out); 

private:

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Begin Luminosity Block
  void dqmBeginLuminosityBlock(DQMStore::IBooker & ibooker , DQMStore::IGetter & igetter , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) ;

  /// End Luminosity Block
  void dqmEndLuminosityBlock(DQMStore::IBooker & ibooker , DQMStore::IGetter & igetter , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup);

  /// EndRun
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Endjob
  void dqmEndJob(DQMStore::IBooker & ibooker_, DQMStore::IGetter & igetter_);



private:

  SiStripClassToMonitorCondData* condDataMon_;  
  void checkTrackerFEDsInLS(DQMStore::IGetter & igetter, double iLS);

  //SiStripWebInterface* sistripWebInterface_;

  int fileSaveFrequency_;
  int summaryFrequency_;
  int tkMapFrequency_;
  int staticUpdateFrequency_;
  int globalStatusFilling_;
  int shiftReportFrequency_;
  bool verbose_;

  edm::InputTag rawDataTag_;
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;

  std::string outputFilePath_;
  std::string outputFileName_;

  edm::ParameterSet tkMapPSet_;
  edm::ESHandle< SiStripFedCabling > fedCabling_;
  edm::ESHandle< SiStripDetCabling > detCabling_;
  SiStripActionExecutor* actionExecutor_;

  unsigned long long m_cacheID_;
  int nLumiSecs_;
  bool trackerFEDsFound_;
  bool printFaultyModuleList_;
  bool endLumiAnalysisOn_;
  std::ostringstream html_out_;
  std::string nFEDinfoDir_;
  std::string nFEDinVsLSname_;
};


#endif
