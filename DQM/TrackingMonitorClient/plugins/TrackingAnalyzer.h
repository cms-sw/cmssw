#ifndef TrackingAnalyser_H
#define TrackingAnalyser_H

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class SiStripFedCabling;
class SiStripDetCabling;
class TrackingActionExecutor;
class FEDRawDataCollection;
class SiStripFedCablingRcd;
class SiStripDetCablingRcd;

class TrackingAnalyser : public DQMEDHarvester {
public:
  /// Constructor
  TrackingAnalyser(const edm::ParameterSet& ps);

  /// Destructor
  ~TrackingAnalyser() override;

private:
  /// BeginJob
  void beginJob() override;

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  /// Begin Luminosity Block
  void dqmBeginLuminosityBlock(DQMStore::IBooker& ibooker_,
                               DQMStore::IGetter& igetter_,
                               edm::LuminosityBlock const& lumiSeg,
                               edm::EventSetup const& eSetup);

  /// End Luminosity Block
  void dqmEndLuminosityBlock(DQMStore::IBooker& ibooker_,
                             DQMStore::IGetter& igetter_,
                             edm::LuminosityBlock const& lumiSeg,
                             edm::EventSetup const& eSetup) override;

  /// Endjob
  void dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) override;

private:
  bool verbose_;

  void checkTrackerFEDsInLS(DQMStore::IGetter& igetter, double iLS);
  void checkTrackerFEDsWdataInLS(DQMStore::IGetter& igetter, double iLS);

  int fileSaveFrequency_;
  int staticUpdateFrequency_;
  int globalStatusFilling_;
  int shiftReportFrequency_;

  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;

  std::string outputFilePath_;
  std::string outputFileName_;

  edm::ParameterSet tkMapPSet_;
  edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;
  edm::ESWatcher<SiStripFedCablingRcd> fedCablingWatcher_;
  const SiStripFedCabling* fedCabling_;
  const SiStripDetCabling* detCabling_;
  TrackingActionExecutor* actionExecutor_;

  unsigned long long m_cacheID_;
  int nLumiSecs_;
  bool trackerFEDsFound_;
  bool trackerFEDsWdataFound_;
  std::ostringstream html_out_;

  std::string nFEDinfoDir_;
  std::string nFEDinVsLSname_;
  std::string nFEDinWdataVsLSname_;
};

#endif
