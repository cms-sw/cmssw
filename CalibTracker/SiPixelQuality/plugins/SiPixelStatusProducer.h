#ifndef CalibTracker_SiPixelQuality_SiPixelStatusProducer_h
#define CalibTracker_SiPixelQuality_SiPixelStatusProducer_h

/**_________________________________________________________________
   class:   SiPixelStatusProducer.h
   package: CalibTracker/SiPixelQuality

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiPixelStatusProducer : public edm::one::EDProducer<edm::EndLuminosityBlockProducer,
                                                         edm::one::WatchLuminosityBlocks, edm::Accumulator> {
 public:
  explicit SiPixelStatusProducer(const edm::ParameterSet&);
  ~SiPixelStatusProducer() override;

 private:
  void beginLuminosityBlock     (edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) final;
  void endLuminosityBlock       (edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) final;
  void endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) final;
  void accumulate                  (edm::Event const&, const edm::EventSetup& iSetup) final;
  
  virtual void onlineRocColRow(const DetId &detId, int offlineRow, int offlineCol, int &roc, int &row, int &col) final;

  virtual int indexROC(int irow, int icol, int nROCcolumns) final;

  // time granularity control
  unsigned long int ftotalevents;
  int resetNLumi_;
  int countLumi_;      //counter

  int beginLumi_;
  int endLumi_;
  int beginRun_;
  int endRun_;

  // condition watchers
  // CablingMaps
  edm::ESWatcher<SiPixelFedCablingMapRcd> siPixelFedCablingMapWatcher_;
  edm::ESHandle<SiPixelFedCablingMap> fCablingMap;
  const SiPixelFedCablingMap* fCablingMap_;

  // TrackerDIGIGeo
  edm::ESWatcher<TrackerDigiGeometryRecord> trackerDIGIGeoWatcher_;
  edm::ESHandle<TrackerGeometry> fTG;
  // TrackerTopology
  edm::ESWatcher<TrackerTopologyRcd> trackerTopoWatcher_;

  // SiPixel offline<->online conversion
  // -- map (for each detid) of the map from offline col/row to the online roc/col/row
  SiPixelCoordinates coord_;

  // ROC size (number of row, number of columns for each det id)
  std::map<int, std::pair<int,int> > fSensors;
  // the roc layout on a module
  std::map<int, std::pair<int,int> > fSensorLayout;
  // fedId as a function of detId
  std::unordered_map<uint32_t, unsigned int> fFedIds;
  // map the index ROC to rocId
  std::map<int, std::map<int,int> >fRocIds;

  // Producer inputs / controls
  edm::InputTag                                           fPixelClusterLabel_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>>  fSiPixelClusterToken_;
  std::vector<edm::EDGetTokenT<PixelFEDChannelCollection> > theBadPixelFEDChannelsTokens_;

  // Producer production (output collection)
  SiPixelDetectorStatus                                    fDet;

};

#endif
