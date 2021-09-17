#ifndef CalibTracker_SiPixelQuality_SiPixelStatusProducer_h
#define CalibTracker_SiPixelQuality_SiPixelStatusProducer_h

/**_________________________________________________________________
 *    class:   SiPixelStatusProducer.h
 *       package: CalibTracker/SiPixelQuality
 *          reference : https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface
 *________________________________________________________________**/

// C++ standard
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

#include <string>
// // CMS FW
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
// Concurrency
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

// // Pixel data format
#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
// Tracker Geo
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
// SiPixelTopoFinder
#include "CalibTracker/SiPixelQuality/interface/SiPixelTopoFinder.h"
// SiPixelDetectorStatus
#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"

/* Cache to pertain SiPixelTopoFinder */
class SiPixelStatusCache {
public:
  //NOTE: these are only changes in the constructor call
  mutable edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  mutable edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  mutable edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> siPixelFedCablingMapToken_;
};

/*|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||*/

class SiPixelStatusProducer :

    public edm::stream::EDProducer<edm::GlobalCache<SiPixelStatusCache>,
                                   edm::RunCache<SiPixelTopoFinder>,
                                   edm::LuminosityBlockSummaryCache<std::vector<SiPixelDetectorStatus>>,
                                   edm::EndLuminosityBlockProducer,
                                   edm::Accumulator> {
public:
  SiPixelStatusProducer(edm::ParameterSet const& iPSet, SiPixelStatusCache const*);
  ~SiPixelStatusProducer() override;

  /* module description */
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    {
      edm::ParameterSetDescription psd0;
      psd0.addUntracked<edm::InputTag>("pixelClusterLabel", edm::InputTag("siPixelClusters", "", "RECO"));
      psd0.add<std::vector<edm::InputTag>>("badPixelFEDChannelCollections",
                                           {
                                               edm::InputTag("siPixelDigis"),
                                           });
      desc.add<edm::ParameterSetDescription>("SiPixelStatusProducerParameters", psd0);
    }
    descriptions.add("siPixelStatusProducer", desc);
  }

  /*|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||*/

  /* For each instance of the module*/
  void beginRun(edm::Run const&, edm::EventSetup const&) final;

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final;

  void accumulate(edm::Event const& iEvent, edm::EventSetup const& iSetup) final;

  void endLuminosityBlockSummary(edm::LuminosityBlock const& iLumi,
                                 edm::EventSetup const&,
                                 std::vector<SiPixelDetectorStatus>* siPixelDetectorStatusVtr) const final;  //override;

  /* For global or runCache */

  static std::unique_ptr<SiPixelStatusCache> initializeGlobalCache(edm::ParameterSet const& iPSet) {
    edm::LogInfo("SiPixelStatusProducer") << "Init global Cache " << std::endl;
    return std::make_unique<SiPixelStatusCache>();
  }

  static std::shared_ptr<SiPixelTopoFinder> globalBeginRun(edm::Run const& iRun,
                                                           edm::EventSetup const& iSetup,
                                                           GlobalCache const* iCache);

  static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
    /* Do nothing */
  }

  static void globalEndJob(SiPixelStatusCache const*) { /* Do nothing */
  }

  static std::shared_ptr<std::vector<SiPixelDetectorStatus>> globalBeginLuminosityBlockSummary(
      edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const*) {
    return std::make_shared<std::vector<SiPixelDetectorStatus>>();
  }

  static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                              edm::EventSetup const&,
                                              LuminosityBlockContext const* iContext,
                                              std::vector<SiPixelDetectorStatus>*) {
    /* Do nothing */
  }

  static void globalEndLuminosityBlockProduce(edm::LuminosityBlock& iLumi,
                                              edm::EventSetup const&,
                                              LuminosityBlockContext const* iContext,
                                              std::vector<SiPixelDetectorStatus> const* siPixelDetectorStatusVtr) {
    edm::LogInfo("SiPixelStatusProducer") << "Global endlumi producer " << std::endl;

    // only save result for non-zero event lumi block
    if (!siPixelDetectorStatusVtr->empty()) {
      int lumi = iLumi.luminosityBlock();
      int run = iLumi.run();

      SiPixelDetectorStatus siPixelDetectorStatus = SiPixelDetectorStatus();
      for (unsigned int instance = 0; instance < siPixelDetectorStatusVtr->size(); instance++) {
        siPixelDetectorStatus.updateDetectorStatus((*siPixelDetectorStatusVtr)[instance]);
      }

      siPixelDetectorStatus.setRunRange(run, run);
      siPixelDetectorStatus.setLSRange(lumi, lumi);

      if (debug_) {
        std::string outTxt = Form("SiPixelDetectorStatus_Run%d_Lumi%d.txt", run, lumi);
        std::ofstream outFile;
        outFile.open(outTxt.c_str(), std::ios::app);
        siPixelDetectorStatus.dumpToFile(outFile);
        outFile.close();
      }

      /* save result */
      auto result = std::make_unique<SiPixelDetectorStatus>();
      *result = siPixelDetectorStatus;

      iLumi.put(std::move(result), std::string("siPixelStatus"));
      edm::LogInfo("SiPixelStatusProducer")
          << " lumi-based data stored for run " << run << " lumi " << lumi << std::endl;
    }
  }

private:
  virtual int indexROC(int irow, int icol, int nROCcolumns) final;

  /* ParameterSet */
  static const bool debug_ = false;

  edm::InputTag fPixelClusterLabel_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> fSiPixelClusterToken_;
  std::vector<edm::EDGetTokenT<PixelFEDChannelCollection>> theBadPixelFEDChannelsTokens_;

  /*|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||*/
  /* private data member, one instance per stream */

  /* per-Run data (The pixel topo cannot be changed during a Run) */
  /* vector of all <int> detIds */
  std::vector<int> fDetIds_;
  /* ROC size (number of row, number of columns for each det id) */
  std::map<int, std::pair<int, int>> fSensors_;
  /* the roc layout on a module */
  std::map<int, std::pair<int, int>> fSensorLayout_;
  /* fedId as a function of detId */
  std::unordered_map<uint32_t, unsigned int> fFedIds_;
  /* map the index ROC to rocId */
  std::map<int, std::map<int, int>> fRocIds_;

  /* per-LuminosityBlock data */
  unsigned long int ftotalevents_;

  int beginLumi_;
  int endLumi_;
  int beginRun_;
  int endRun_;

  /* Channels always have FEDerror25 for all events in the lumisection */
  std::map<int, std::vector<PixelFEDChannel>> fFEDerror25_;

  // Producer production (output collection)
  SiPixelDetectorStatus fDet_;
};

#endif
