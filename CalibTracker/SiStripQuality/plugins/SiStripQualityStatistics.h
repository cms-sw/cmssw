#ifndef SiStripQualityStatistics_H
#define SiStripQualityStatistics_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

class SiStripDetInfoFileReader;
class SiStripFedCabling;

class SiStripQualityStatistics : public DQMEDHarvester {
public:
  explicit SiStripQualityStatistics(const edm::ParameterSet&);
  ~SiStripQualityStatistics() override;

  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  void updateAndSave(const SiStripQuality* siStripQuality);
  void SetBadComponents(int, int, SiStripQuality::BadComponent&);

  unsigned long long m_cacheID_;
  edm::RunID run_;
  std::string dataLabel_;
  std::string TkMapFileName_;
  bool saveTkHistoMap_;
  //Global Info
  int NTkBadComponent[4];  //k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  int NBadComponent[4][19][4];
  //legend: NBadComponent[i][j][k]= SubSystem i, layer/disk/wheel j, BadModule/Fiber/Apv k
  //     i: 0=TIB, 1=TID, 2=TOB, 3=TEC
  //     k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  std::stringstream ssV[4][19];

  TrackerMap *tkMap, *tkMapFullIOVs;
  SiStripDetInfoFileReader* reader;
  std::unique_ptr<TkHistoMap> tkhisto;
  bool addBadCompFromFedErr_;
  float fedErrCutoff_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;
  edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
  edm::ESWatcher<SiStripQualityRcd> stripQualityWatcher_;
  const TrackerTopology* tTopo_;
  const SiStripFedCabling* fedCabling_;
  const SiStripQuality* siStripQuality_;
};
#endif
