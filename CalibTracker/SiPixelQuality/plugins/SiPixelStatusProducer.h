#ifndef CalibTracker_SiPixelQuality_SiPixelStatusProducer_h
#define CalibTracker_SiPixelQuality_SiPixelStatusProducer_h

/**_________________________________________________________________
   class:   SiPixelStatusProducer.h
   package: CalibTracker/SiPixelQuality

________________________________________________________________**/

// C++ standard
#include <string>
// CMS FW
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

// Pixel data format
#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
// Tracker Geo
#include "DQM/SiPixelPhase1Common/interface/SiPixelCoordinates.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

class SiPixelStatusProducer
    : public edm::one::EDProducer<edm::EndLuminosityBlockProducer, edm::one::WatchLuminosityBlocks, edm::Accumulator> {
public:
  explicit SiPixelStatusProducer(const edm::ParameterSet&);
  ~SiPixelStatusProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup&) final;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup&) final;
  void endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup&) final;
  void accumulate(edm::Event const&, const edm::EventSetup&) final;

  virtual void onlineRocColRow(const DetId& detId, int offlineRow, int offlineCol, int& roc, int& row, int& col) final;

  virtual int indexROC(int irow, int icol, int nROCcolumns) final;

  // time granularity control
  unsigned long int ftotalevents;
  int resetNLumi_;
  int countLumi_;  //counter

  int beginLumi_;
  int endLumi_;
  int beginRun_;
  int endRun_;

  // condition watchers
  // CablingMaps
  edm::ESWatcher<SiPixelFedCablingMapRcd> siPixelFedCablingMapWatcher_;
  const SiPixelFedCablingMap* fCablingMap_ = nullptr;

  // TrackerDIGIGeo
  edm::ESWatcher<TrackerDigiGeometryRecord> trackerDIGIGeoWatcher_;
  const TrackerGeometry* trackerGeometry_ = nullptr;

  // TrackerTopology
  edm::ESWatcher<TrackerTopologyRcd> trackerTopoWatcher_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> siPixelFedCablingMapToken_;

  // SiPixel offline<->online conversion
  // -- map (for each detid) of the map from offline col/row to the online roc/col/row
  SiPixelCoordinates coord_;

  // ROC size (number of row, number of columns for each det id)
  std::map<int, std::pair<int, int>> fSensors;
  // the roc layout on a module
  std::map<int, std::pair<int, int>> fSensorLayout;
  // fedId as a function of detId
  std::unordered_map<uint32_t, unsigned int> fFedIds;
  // map the index ROC to rocId
  std::map<int, std::map<int, int>> fRocIds;

  // Producer inputs / controls
  edm::InputTag fPixelClusterLabel_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> fSiPixelClusterToken_;
  std::vector<edm::EDGetTokenT<PixelFEDChannelCollection>> theBadPixelFEDChannelsTokens_;

  // Channels always have FEDerror25 for the full lumi section
  std::map<int, std::vector<PixelFEDChannel>> FEDerror25_;

  // Producer production (output collection)
  SiPixelDetectorStatus fDet;
};

#endif
