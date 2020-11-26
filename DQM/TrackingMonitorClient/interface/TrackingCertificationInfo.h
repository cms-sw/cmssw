#ifndef TrackingMonitorClient_TrackingCertificationInfo_h
#define TrackingMonitorClient_TrackingCertificationInfo_h
// -*- C++ -*-
//
// Package:     TrackingMonitorClient
// Class  :     TrackingCertificationInfo
//
/**\class TrackingCertificationInfo TrackingCertificationInfo.h DQM/TrackingMonitorClient/interface/TrackingCertificationInfo.h

 Description: 

 Usage:
    <usage>

*/

#include <string>

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <fstream>
#include <string>
#include <vector>
#include <map>

class SiStripDetCabling;
class SiStripDetCablingRcd;
class RunInfo;
class RunInfoRcd;

class TrackingCertificationInfo : public DQMEDHarvester {
public:
  /// Constructor
  TrackingCertificationInfo(const edm::ParameterSet& ps);

  /// Destructor
  ~TrackingCertificationInfo() override;

private:
  /// BeginJob
  void beginJob() override;

  /// Begin Run
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  /// End Of Luminosity
  void dqmEndLuminosityBlock(DQMStore::IBooker& ibooker_,
                             DQMStore::IGetter& igetter_,
                             edm::LuminosityBlock const& lumiSeg,
                             edm::EventSetup const& iSetup) override;

  /// EndJob
  void dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) override;

private:
  void bookTrackingCertificationMEs(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_);
  void bookTrackingCertificationMEsAtLumi(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_);

  void resetTrackingCertificationMEs(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_);
  void resetTrackingCertificationMEsAtLumi(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_);

  void fillTrackingCertificationMEs(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_);
  void fillTrackingCertificationMEsAtLumi(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_);

  void fillDummyTrackingCertification(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_);
  void fillDummyTrackingCertificationAtLumi(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_);

  struct TrackingMEs {
    MonitorElement* TrackingFlag;
  };

  struct TrackingLSMEs {
    MonitorElement* TrackingFlag;
  };

  std::map<std::string, TrackingMEs> TrackingMEsMap;
  std::map<std::string, TrackingLSMEs> TrackingLSMEsMap;

  MonitorElement* TrackingCertification;
  MonitorElement* TrackingCertificationSummaryMap;

  MonitorElement* TrackingLSCertification;

  edm::ParameterSet pSet_;

  bool trackingCertificationBooked_;
  bool trackingLSCertificationBooked_;
  int nFEDConnected_;
  bool allPixelFEDConnected_;
  bool verbose_;
  std::string TopFolderName_;

  bool checkPixelFEDs_;

  unsigned long long m_cacheID_;

  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
  const RunInfo* sumFED_ = nullptr;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;
  edm::ESWatcher<SiStripDetCablingRcd> fedDetCablingWatcher_;
  const SiStripDetCabling* detCabling_;

  std::vector<std::string> SubDetFolder;
};
#endif
