/**_________________________________________________________________
   class:   SiPixelStatusProducer.cc
   package: CalibTracker/SiPixelQuality

   ________________________________________________________________**/

// C++ standard
#include <string>
// ROOT
#include "TMath.h"
// CMSSW FW
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Transition.h"
// CMSSW DataFormats
#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
// "FED error 25"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
// CMSSW CondFormats
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
// EDProducer related dataformat
#include "DQM/SiPixelPhase1Common/interface/SiPixelCoordinates.h"
#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"
// header file
#include "CalibTracker/SiPixelQuality/plugins/SiPixelStatusProducer.h"

using namespace std;

//--------------------------------------------------------------------------------------------------
SiPixelStatusProducer::SiPixelStatusProducer(const edm::ParameterSet& iConfig)
    : trackerGeometryToken_(
          esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginLuminosityBlock>()),
      trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginLuminosityBlock>()),
      siPixelFedCablingMapToken_(
          esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd, edm::Transition::BeginLuminosityBlock>()) {
  // get parameter

  // badPixelFEDChannelCollections
  std::vector<edm::InputTag> badPixelFEDChannelCollectionLabels_ =
      iConfig.getParameter<edm::ParameterSet>("SiPixelStatusProducerParameters")
          .getParameter<std::vector<edm::InputTag>>("badPixelFEDChannelCollections");
  for (auto& t : badPixelFEDChannelCollectionLabels_)
    theBadPixelFEDChannelsTokens_.push_back(consumes<PixelFEDChannelCollection>(t));
  //  badPixelFEDChannelCollections = cms.VInputTag(cms.InputTag('siPixelDigis'))

  fPixelClusterLabel_ = iConfig.getParameter<edm::ParameterSet>("SiPixelStatusProducerParameters")
                            .getUntrackedParameter<edm::InputTag>("pixelClusterLabel");
  fSiPixelClusterToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(fPixelClusterLabel_);
  resetNLumi_ = iConfig.getParameter<edm::ParameterSet>("SiPixelStatusProducerParameters")
                    .getUntrackedParameter<int>("resetEveryNLumi", 1);

  ftotalevents = 0;
  countLumi_ = 0;

  beginLumi_ = endLumi_ = -1;
  endLumi_ = endRun_ = -1;

  produces<SiPixelDetectorStatus, edm::Transition::EndLuminosityBlock>("siPixelStatus");
}

//--------------------------------------------------------------------------------------------------
SiPixelStatusProducer::~SiPixelStatusProducer() {}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {
  edm::LogInfo("SiPixelStatusProducer") << "beginlumi setup " << endl;

  if (countLumi_ == 0 && resetNLumi_ > 0) {
    beginLumi_ = lumiSeg.luminosityBlock();
    beginRun_ = lumiSeg.run();
    ftotalevents = 0;
  }

  // The es watcher is acutally not needed if run parallel jobs for each lumi section
  if (siPixelFedCablingMapWatcher_.check(iSetup) || trackerDIGIGeoWatcher_.check(iSetup) ||
      trackerTopoWatcher_.check(iSetup)) {
    trackerGeometry_ = &iSetup.getData(trackerGeometryToken_);
    const TrackerTopology* trackerTopology = &iSetup.getData(trackerTopologyToken_);
    fCablingMap_ = &iSetup.getData(siPixelFedCablingMapToken_);

    coord_.init(trackerTopology, trackerGeometry_, fCablingMap_);

    fFedIds = fCablingMap_->det2fedMap();

  }  // if conditionWatcher_.check(iSetup)

  // init the SiPixelDetectorStatus fDet and sensor size fSensors in the begining (when countLumi is zero)
  if (countLumi_ == 0) {
    for (TrackerGeometry::DetContainer::const_iterator it = trackerGeometry_->dets().begin();
         it != trackerGeometry_->dets().end();
         it++) {
      const PixelGeomDetUnit* pgdu = dynamic_cast<const PixelGeomDetUnit*>((*it));
      if (pgdu == nullptr)
        continue;
      DetId detId = (*it)->geographicalId();
      int detid = detId.rawId();

      // don't want to use magic number row 80 column 52
      const PixelTopology* topo = static_cast<const PixelTopology*>(&pgdu->specificTopology());
      int rowsperroc = topo->rowsperroc();
      int colsperroc = topo->colsperroc();

      int nROCrows = pgdu->specificTopology().nrows() / rowsperroc;
      int nROCcolumns = pgdu->specificTopology().ncolumns() / colsperroc;
      int nrocs = nROCrows * nROCcolumns;

      fDet.addModule(detid, nrocs);

      fSensors[detid] = std::make_pair(rowsperroc, colsperroc);
      fSensorLayout[detid] = std::make_pair(nROCrows, nROCcolumns);

      std::map<int, int> rocIdMap;
      for (int irow = 0; irow < nROCrows; irow++) {
        for (int icol = 0; icol < nROCcolumns; icol++) {
          int dummyOfflineRow = (rowsperroc / 2 - 1) + irow * rowsperroc;
          int dummeOfflineColumn = (colsperroc / 2 - 1) + icol * colsperroc;
          // encode irow, icol
          int key = indexROC(irow, icol, nROCcolumns);

          int roc(-1), rocR(-1), rocC(-1);
          onlineRocColRow(detId, dummyOfflineRow, dummeOfflineColumn, roc, rocR, rocC);

          int value = roc;
          rocIdMap[key] = value;
        }
      }

      fRocIds[detid] = rocIdMap;
    }

  }  // init when countLumi = 0

  FEDerror25_.clear();
  countLumi_++;
}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusProducer::accumulate(edm::Event const& iEvent, const edm::EventSetup&) {
  ftotalevents++;

  edm::LogInfo("SiPixelStatusProducer") << "start cluster analyzer " << endl;

  // ----------------------------------------------------------------------
  // -- Pixel cluster analysis
  // ----------------------------------------------------------------------

  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> hClusterColl;
  if (!iEvent.getByToken(fSiPixelClusterToken_, hClusterColl)) {
    edm::LogWarning("SiPixelStatusProducer")
        << " edmNew::DetSetVector<SiPixelCluster> " << fPixelClusterLabel_ << " does not exist!" << endl;
    return;
  }

  iEvent.getByToken(fSiPixelClusterToken_, hClusterColl);

  if (hClusterColl.isValid()) {
    for (const auto& clusters : *hClusterColl) {  //loop over different clusters in a clusters vector (module)
      for (const auto& clu : clusters) {          // loop over cluster in a given detId (module)
        int detid = clusters.detId();
        int rowsperroc = fSensors[detid].first;
        int colsperroc = fSensors[detid].second;

        int nROCcolumns = fSensorLayout[detid].second;

        int roc(-1);
        std::map<int, int> fRocIds_detid;
        if (fRocIds.find(detid) != fRocIds.end()) {
          fRocIds_detid = fRocIds[detid];
        }

        const vector<SiPixelCluster::Pixel>& pixvector = clu.pixels();
        for (unsigned int i = 0; i < pixvector.size(); ++i) {
          int mr0 = pixvector[i].x;  // constant column direction is along x-axis,
          int mc0 = pixvector[i].y;  // constant row direction is along y-axis

          int irow = mr0 / rowsperroc;
          int icol = mc0 / colsperroc;

          int key = indexROC(irow, icol, nROCcolumns);
          if (fRocIds_detid.find(key) != fRocIds_detid.end()) {
            roc = fRocIds_detid[key];
          }

          fDet.fillDIGI(detid, roc);

        }  // loop over pixels in a given cluster

      }  // loop over cluster in a given detId (module)

    }  // loop over detId-grouped clusters in cluster detId-grouped clusters-vector

  }  // hClusterColl.isValid()
  else {
    edm::LogWarning("SiPixelStatusProducer")
        << " edmNew::DetSetVector<SiPixelCluster> " << fPixelClusterLabel_ << " is NOT Valid!" << endl;
  }
  //////////////////////////////////////////////////////////////////////

  // FEDerror25 per-ROC per-event
  edm::Handle<PixelFEDChannelCollection> pixelFEDChannelCollectionHandle;

  // look over different resouces of takens
  for (const edm::EDGetTokenT<PixelFEDChannelCollection>& tk : theBadPixelFEDChannelsTokens_) {
    // collection has to exist
    if (!iEvent.getByToken(tk, pixelFEDChannelCollectionHandle)) {
      edm::LogWarning("SiPixelStatusProducer")
          << " PixelFEDChannelCollection with index " << tk.index() << " does NOT exist!" << std::endl;
      continue;
    }
    iEvent.getByToken(tk, pixelFEDChannelCollectionHandle);
    // collection has to be valid
    if (!pixelFEDChannelCollectionHandle.isValid()) {
      edm::LogWarning("SiPixelStatusProducer")
          << " PixelFEDChannelCollection with index " << tk.index() << " is NOT valid!" << endl;
      continue;
    }
    // FEDerror channels for the current events
    std::map<int, std::vector<PixelFEDChannel>> tmpFEDerror25;
    for (const auto& disabledChannels : *pixelFEDChannelCollectionHandle) {
      //loop over different PixelFED in a PixelFED vector (module)
      for (const auto& ch : disabledChannels) {
        DetId detId = disabledChannels.detId();
        int detid = detId.rawId();

        if (ftotalevents == 1) {
          // FEDerror25 channels for the first event in the lumi section
          FEDerror25_[detid].push_back(ch);
        } else
          tmpFEDerror25[detid].push_back(ch);

      }  // loop over different PixelFED in a PixelFED vector (different channel for a given module)

    }  // loop over different (different DetId) PixelFED vectors in PixelFEDChannelCollection

    // Compare the current FEDerror list with the first event's FEDerror list
    // and save the common channels
    if (!tmpFEDerror25.empty() && !FEDerror25_.empty()) {  // non-empty FEDerror lists

      std::map<int, std::vector<PixelFEDChannel>>::iterator itFEDerror25;
      for (itFEDerror25 = FEDerror25_.begin(); itFEDerror25 != FEDerror25_.end(); itFEDerror25++) {
        int detid = itFEDerror25->first;
        if (tmpFEDerror25.find(detid) != tmpFEDerror25.end()) {
          std::vector<PixelFEDChannel> chs = itFEDerror25->second;
          std::vector<PixelFEDChannel> chs_tmp = tmpFEDerror25[detid];

          std::vector<PixelFEDChannel> chs_common;
          for (unsigned int ich = 0; ich < chs.size(); ich++) {
            PixelFEDChannel ch = chs[ich];
            // look over the current FEDerror25 channels, save the common FED channels
            for (unsigned int ich_tmp = 0; ich_tmp < chs_tmp.size(); ich_tmp++) {
              PixelFEDChannel ch_tmp = chs_tmp[ich_tmp];
              if ((ch.fed == ch_tmp.fed) && (ch.link == ch_tmp.link)) {  // the same FED channel
                chs_common.push_back(ch);
                break;
              }
            }
          }
          // remove the full module from FEDerror25 list if no common channels are left
          if (chs_common.empty())
            FEDerror25_.erase(itFEDerror25);
          // otherwise replace with the common channels
          else {
            FEDerror25_[detid].clear();
            FEDerror25_[detid] = chs_common;
          }
        } else {  // remove the full module from FEDerror25 list if the module doesn't appear in the current event's FEDerror25 list
          FEDerror25_.erase(itFEDerror25);
        }

      }  // loop over modules that have FEDerror25 in the first event in the lumi section

    }  // non-empty FEDerror lists

  }  // look over different resouces of takens

  // no per-event collection put into iEvent
  // If use produce() function and no collection is put into iEvent, produce() will not run in unScheduled mode
  // Now since CMSSW_10_1_X, the accumulate() function will run whatsoever in the unScheduled mode
  // But accumulate() is NOT available and will NOT be available for releases before CMSSW_10_1_X
}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup&) {}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusProducer::endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup&) {
  edm::LogInfo("SiPixelStatusProducer") << "endlumi producer " << endl;

  endLumi_ = lumiSeg.luminosityBlock();
  endRun_ = lumiSeg.run();

  // check if countLumi_ is large enough to read out/save data and reset for the next round
  if (resetNLumi_ == -1)
    return;
  if (countLumi_ < resetNLumi_)
    return;

  // set the FEDerror25 flag to be true for ROCs send out FEDerror25 for all events in the lumi section
  if (!FEDerror25_.empty()) {
    std::map<int, std::vector<PixelFEDChannel>>::iterator itFEDerror25;
    for (itFEDerror25 = FEDerror25_.begin(); itFEDerror25 != FEDerror25_.end(); itFEDerror25++) {
      int detid = itFEDerror25->first;
      std::vector<PixelFEDChannel> chs = itFEDerror25->second;
      for (unsigned int ich = 0; ich < chs.size(); ich++) {
        fDet.fillFEDerror25(detid, chs[ich]);
      }
    }
  }

  fDet.setRunRange(beginRun_, endRun_);
  fDet.setLSRange(beginLumi_, endLumi_);
  fDet.setNevents(ftotalevents);

  // save result
  auto result = std::make_unique<SiPixelDetectorStatus>();
  *result = fDet;

  // only save for the lumi sections with NON-ZERO events
  lumiSeg.put(std::move(result), std::string("siPixelStatus"));
  edm::LogInfo("SiPixelStatusProducer") << "new lumi-based data stored for run " << beginRun_ << " lumi from "
                                        << beginLumi_ << " to " << endLumi_ << std::endl;

  // reset detector status and lumi-counter
  fDet.resetDetectorStatus();
  countLumi_ = 0;
  ftotalevents = 0;
  FEDerror25_.clear();
}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusProducer::onlineRocColRow(
    const DetId& detId, int offlineRow, int offlineCol, int& roc, int& row, int& col) {
  int fedId = fFedIds[detId.rawId()];

  // from detector to cabling
  sipixelobjects::ElectronicIndex cabling;
  sipixelobjects::DetectorIndex detector;  //{detId.rawId(), offlineRow, offlineCol};
  detector.rawId = detId.rawId();
  detector.row = offlineRow;
  detector.col = offlineCol;

  SiPixelFrameConverter converter(fCablingMap_, fedId);
  converter.toCabling(cabling, detector);

  // then one can construct local pixel
  sipixelobjects::LocalPixel::DcolPxid loc;
  loc.dcol = cabling.dcol;
  loc.pxid = cabling.pxid;
  // and get local(online) row/column
  sipixelobjects::LocalPixel locpixel(loc);
  col = locpixel.rocCol();
  row = locpixel.rocRow();
  //sipixelobjects::CablingPathToDetUnit path = {(unsigned int) fedId, (unsigned int)cabling.link, (unsigned int)cabling.roc};
  //const sipixelobjects::PixelROC *theRoc = fCablingMap_->findItem(path);
  const sipixelobjects::PixelROC* theRoc = converter.toRoc(cabling.link, cabling.roc);
  roc = theRoc->idInDetUnit();

  // has to be BPIX; has to be minus side; has to be half module
  // for phase-I, there is no half module
  if (detId.subdetId() == PixelSubdetector::PixelBarrel && coord_.side(detId) == 1 && coord_.half(detId)) {
    roc += 8;
  }
}

//--------------------------------------------------------------------------------------------------
int SiPixelStatusProducer::indexROC(int irow, int icol, int nROCcolumns) {
  return int(icol + irow * nROCcolumns);

  // generate the folling roc index that is going to map with ROC id as
  // 8  9  10 11 12 13 14 15
  // 0  1  2  3  4  5  6  7
}

//--------------------------------------------------------------------------------------------------
//edmPythonConfigToCppValidation CalibTracker/SiPixelQuality/python/SiPixelStatusProducer_cfi.py
void SiPixelStatusProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // siPixelStatusProducer
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.addUntracked<int>("resetEveryNLumi", 1);
    psd0.addUntracked<edm::InputTag>("pixelClusterLabel", edm::InputTag("siPixelClusters", "", "RECO"));
    psd0.add<std::vector<edm::InputTag>>("badPixelFEDChannelCollections",
                                         {
                                             edm::InputTag("siPixelDigis"),
                                         });
    desc.add<edm::ParameterSetDescription>("SiPixelStatusProducerParameters", psd0);
  }
  descriptions.add("siPixelStatusProducer", desc);
}

DEFINE_FWK_MODULE(SiPixelStatusProducer);
