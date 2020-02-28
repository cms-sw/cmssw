#ifndef SiStripMonitorClient_SiStripBadComponentInfo_h
#define SiStripMonitorClient_SiStripBadComponentInfo_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripBadComponentInfo
//
/**\class SiStripBadComponentInfo SiStripBadComponentInfo.h
 DQM/SiStripMonitorCluster/interface/SiStripBadComponentInfo.h

 Description:
      Checks the # of SiStrip FEDs from DAQ
 Usage:
    <usage>

*/
//
//          Author:  Suchandra Dutta
//         Created:  Fri Jan 26 10:00:00 CET 2018
//

#include <string>

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class SiStripBadComponentInfo : public DQMEDHarvester {
public:
  /// Constructor
  SiStripBadComponentInfo(edm::ParameterSet const& ps);
  ~SiStripBadComponentInfo() override;

protected:
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker&,
                 DQMStore::IGetter&) override;  // performed in the endJob

private:
  void checkBadComponents(edm::EventSetup const& eSetup);
  void bookBadComponentHistos(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
  void fillBadComponentMaps(const SiStripQuality* siStripQuality);
  void fillBadComponentMaps(int xbin, int component, SiStripQuality::BadComponent const& BC);
  void createSummary(MonitorElement* me, const std::map<std::pair<int, int>, float>& map);

  MonitorElement* badAPVME_;
  MonitorElement* badFiberME_;
  MonitorElement* badStripME_;

  std::map<std::pair<int, int>, float> mapBadAPV;
  std::map<std::pair<int, int>, float> mapBadFiber;
  std::map<std::pair<int, int>, float> mapBadStrip;

  bool bookedStatus_;
  int nSubSystem_;
  std::string qualityLabel_;

  edm::ESHandle<SiStripQuality> siStripQuality_;
  edm::ESHandle<TrackerTopology> tTopo_;
  edm::ESHandle<SiStripFedCabling> fedCabling_;
  bool addBadCompFromFedErr_;
  float fedErrCutoff_;
};
#endif
