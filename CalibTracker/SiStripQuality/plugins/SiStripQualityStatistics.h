#ifndef SiStripQualityStatistics_H
#define SiStripQualityStatistics_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "CalibTracker/SiStripQuality/interface/SiStripQualityWithFromFedErrorsHelper.h"

class SiStripFedCabling;

class SiStripQualityStatistics : public DQMEDHarvester {
public:
  explicit SiStripQualityStatistics(const edm::ParameterSet&);
  ~SiStripQualityStatistics() override = default;

  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void updateAndSave(const SiStripQuality* siStripQuality);
  void SetBadComponents(int, int, SiStripQuality::BadComponent&);

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
  SiStripDetInfo detInfo_;
  std::unique_ptr<TkHistoMap> tkhisto;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_;
  std::unique_ptr<TrackerTopology> tTopo_;
  SiStripQualityWithFromFedErrorsHelper withFedErrHelper_;
};
#endif
