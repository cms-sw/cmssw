#ifndef SiStripAnalyser_H
#define SiStripAnalyser_H

/** \class SiStripAnalyser
 * *
 *  SiStrip SiStripAnalyser
 *  \author  S. Dutta INFN-Pisa
 *
 */

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripClassToMonitorCondData.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <map>
#include <string>
#include <vector>

class SiStripWebInterface;
class SiStripFedCabling;
class SiStripDetCabling;
class SiStripClassToMonitorCondData;
class FEDRawDataCollection;

class SiStripAnalyser : public edm::EDAnalyzer {
public:
  SiStripAnalyser(const edm::ParameterSet& ps);
  ~SiStripAnalyser() override;

private:
  void beginJob() override;
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) override;
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void endJob() override;

  void checkTrackerFEDs(edm::Event const& e);

  SiStripClassToMonitorCondData condDataMon_;
  SiStripActionExecutor actionExecutor_;
  edm::ParameterSet tkMapPSet_;

  int summaryFrequency_{-1};
  int staticUpdateFrequency_;
  int globalStatusFilling_;
  int shiftReportFrequency_;

  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;

  std::string outputFilePath_;
  std::string outputFileName_;

  edm::ESHandle<SiStripFedCabling> fedCabling_;
  edm::ESHandle<SiStripDetCabling> detCabling_;

  unsigned long long m_cacheID_;
  int nLumiSecs_{};
  int nEvents_{};
  bool trackerFEDsFound_{false};
  bool printFaultyModuleList_;
  bool endLumiAnalysisOn_{false};
};

#endif
