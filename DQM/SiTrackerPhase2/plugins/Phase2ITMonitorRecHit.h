#ifndef _Validation_SiTrackerPhase2V_Phase2ITMonitorRecHit_h
#define _Validation_SiTrackerPhase2V_Phase2ITMonitorRecHit_h
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

// DQM Histograming
class TrackerTopology;
class TrackerGeometry;
class TrackerDigiGeometryRecord;
class TrackerTopologyRcd;
class Phase2ITMonitorRecHit : public DQMEDAnalyzer {
public:
  explicit Phase2ITMonitorRecHit(const edm::ParameterSet&);
  ~Phase2ITMonitorRecHit() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void fillITHistos(const edm::Event& iEvent);

  void bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir);

  edm::ParameterSet config_;
  std::string geomType_;
  const edm::EDGetTokenT<SiPixelRecHitCollection> tokenRecHitsIT_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;

  MonitorElement* numberRecHits_;
  MonitorElement* globalXY_barrel_;
  MonitorElement* globalXY_endcap_;
  MonitorElement* globalRZ_barrel_;
  MonitorElement* globalRZ_endcap_;

  struct RecHitME {
    MonitorElement* numberRecHits = nullptr;
    MonitorElement* globalPosXY = nullptr;
    MonitorElement* globalPosRZ = nullptr;
    MonitorElement* localPosXY = nullptr;
    MonitorElement* posX = nullptr;
    MonitorElement* posY = nullptr;
    MonitorElement* poserrX = nullptr;
    MonitorElement* poserrY = nullptr;
    MonitorElement* clusterSizeX = nullptr;
    MonitorElement* clusterSizeY = nullptr;
  };
  std::map<std::string, RecHitME> layerMEs_;
};
#endif
