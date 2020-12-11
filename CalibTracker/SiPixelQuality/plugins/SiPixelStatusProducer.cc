// C++ standard
#include <string>
// ROOT
#include "TMath.h"
// CMSSW FW
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// CMSSW DataFormats
#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
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

// Header file
#include "CalibTracker/SiPixelQuality/plugins/SiPixelStatusProducer.h"

SiPixelStatusProducer::SiPixelStatusProducer(const edm::ParameterSet& iConfig, SiPixelTopoCache const*) {
  /* badPixelFEDChannelCollections */
  std::vector<edm::InputTag> badPixelFEDChannelCollectionLabels =
      iConfig.getParameter<edm::ParameterSet>("SiPixelStatusProducerParameters")
          .getParameter<std::vector<edm::InputTag>>("badPixelFEDChannelCollections");
  for (auto& t : badPixelFEDChannelCollectionLabels)
    theBadPixelFEDChannelsTokens_.push_back(consumes<PixelFEDChannelCollection>(t));

  /* pixel clusters */
  fPixelClusterLabel_ = iConfig.getParameter<edm::ParameterSet>("SiPixelStatusProducerParameters")
                            .getUntrackedParameter<edm::InputTag>("pixelClusterLabel");
  fSiPixelClusterToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(fPixelClusterLabel_);

  //debug_ = iConfig.getUntrackedParameter<bool>("debug");

  /* register products */
  produces<SiPixelDetectorStatus, edm::Transition::EndLuminosityBlock>("siPixelStatus");
}

SiPixelStatusProducer::~SiPixelStatusProducer() {}

//--------------------------------------------------------------------------------------------------

void SiPixelStatusProducer::beginRun(edm::Run const&, edm::EventSetup const&) {
  /*Is it good to pass the objects stored in runCache to set class private members values  *
     or just call runCahche every time in the calss function?*/

  edm::LogInfo("SiPixelStatusProducer") << "beginRun: update the std::map for pixel geo/topo " << std::endl;
  /* update the std::map for pixel geo/topo */
  /* vector of all <int> detIds */
  fDetIds_ = runCache()->getDetIds();  //getDetIds();
  /* ROC size (number of row, number of columns for each det id) */
  fSensors_ = runCache()->getSensors();
  /* the roc layout on a module */
  fSensorLayout_ = runCache()->getSensorLayout();
  /* fedId as a function of detId */
  fFedIds_ = runCache()->getFedIds();
  /* map the index ROC to rocId */
  fRocIds_ = runCache()->getRocIds();
}

void SiPixelStatusProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
  edm::LogInfo("SiPixelStatusProducer") << "beginlumi instance" << std::endl;

  /* initialize fDet_ with a set of modules(detIds) and clean the fFEDerror25_ */
  fDet_ = SiPixelDetectorStatus();
  for (unsigned int itDetId = 0; itDetId < fDetIds_.size(); ++itDetId) {
    int detid = fDetIds_[itDetId];
    int nrocs = fSensorLayout_[detid].first * fSensorLayout_[detid].second;

    fDet_.addModule(detid, nrocs);
  }

  fFEDerror25_.clear();
  ftotalevents_ = 0;
}

void SiPixelStatusProducer::accumulate(edm::Event const& iEvent, edm::EventSetup const&) {
  edm::LogInfo("SiPixelStatusProducer") << "start cluster analyzer " << std::endl;

  /* count number of events for the current module instance in the luminosityBlock */
  ftotalevents_++;

  /* ----------------------------------------------------------------------
     -- Pixel cluster analysis
     ----------------------------------------------------------------------*/

  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> hClusterColl;
  if (!iEvent.getByToken(fSiPixelClusterToken_, hClusterColl)) {
    edm::LogWarning("SiPixelStatusProducer")
        << " edmNew::DetSetVector<SiPixelCluster> " << fPixelClusterLabel_ << " does not exist!" << std::endl;
    return;
  }

  iEvent.getByToken(fSiPixelClusterToken_, hClusterColl);

  if (hClusterColl.isValid()) {
    for (const auto& clusters : *hClusterColl) { /*loop over different clusters in a clusters vector (module)*/
      for (const auto& clu : clusters) {         /*loop over cluster in a given detId (module)*/
        int detid = clusters.detId();
        int rowsperroc = fSensors_[detid].first;
        int colsperroc = fSensors_[detid].second;

        //int nROCrows    = fSensorLayout_[detid].first;
        int nROCcolumns = fSensorLayout_[detid].second;

        int roc(-1);
        std::map<int, int> rocIds_detid;
        if (fRocIds_.find(detid) != fRocIds_.end()) {
          rocIds_detid = fRocIds_[detid];
        }

        /* A module is made with a few ROCs
           Need to convert global row/column (on a module) to local row/column (on a ROC) */
        const std::vector<SiPixelCluster::Pixel>& pixvector = clu.pixels();
        for (unsigned int i = 0; i < pixvector.size(); ++i) {
          int mr0 = pixvector[i].x; /* constant column direction is along x-axis */
          int mc0 = pixvector[i].y; /* constant row direction is along y-axis */

          int irow = mr0 / rowsperroc;
          int icol = mc0 / colsperroc;

          int key = indexROC(irow, icol, nROCcolumns);
          if (rocIds_detid.find(key) != rocIds_detid.end()) {
            roc = rocIds_detid[key];
          }

          fDet_.fillDIGI(detid, roc);

        } /* loop over pixels in a cluster */

      } /* loop over cluster in a detId (module) */

    } /* loop over detId-grouped clusters in cluster detId-grouped clusters-vector* */

  } /* hClusterColl.isValid() */
  else {
    edm::LogWarning("SiPixelStatusProducer")
        << " edmNew::DetSetVector<SiPixelCluster> " << fPixelClusterLabel_ << " is NOT Valid!" << std::endl;
  }

  /*|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||*/

  /* FEDerror25 per-ROC per-event */
  edm::Handle<PixelFEDChannelCollection> pixelFEDChannelCollectionHandle;

  /* look over different resouces of tokens */
  for (const edm::EDGetTokenT<PixelFEDChannelCollection>& tk : theBadPixelFEDChannelsTokens_) {
    if (!iEvent.getByToken(tk, pixelFEDChannelCollectionHandle)) {
      edm::LogWarning("SiPixelStatusProducer")
          << " PixelFEDChannelCollection with index " << tk.index() << " does NOT exist!" << std::endl;
      continue;
    }

    iEvent.getByToken(tk, pixelFEDChannelCollectionHandle);
    if (!pixelFEDChannelCollectionHandle.isValid()) {
      edm::LogWarning("SiPixelStatusProducer")
          << " PixelFEDChannelCollection with index " << tk.index() << " is NOT valid!" << std::endl;
      continue;
    }

    /* FEDerror channels for the current events */
    std::map<int, std::vector<PixelFEDChannel>> tmpFEDerror25;
    for (const auto& disabledChannels : *pixelFEDChannelCollectionHandle) {
      /* loop over different PixelFED in a PixelFED vector (module) */
      for (const auto& ch : disabledChannels) {
        DetId detId = disabledChannels.detId();
        int detid = detId.rawId();

        if (ftotalevents_ == 1) {
          /* FEDerror25 channels for the "first" event in the lumi section (first for the current instance of the module) */
          fFEDerror25_[detid].push_back(ch);
        } else
          tmpFEDerror25[detid].push_back(ch);

      } /* loop over different PixelFED in a PixelFED vector (different channel for a module) */

    } /* loop over different (different DetId) PixelFED vectors in PixelFEDChannelCollection */

    /* Compare the current FEDerror list with the first event's FEDerror list
 *        and save the common channels */
    if (!tmpFEDerror25.empty() && !fFEDerror25_.empty()) {
      std::map<int, std::vector<PixelFEDChannel>>::iterator itFEDerror25;
      for (itFEDerror25 = fFEDerror25_.begin(); itFEDerror25 != fFEDerror25_.end(); itFEDerror25++) {
        int detid = itFEDerror25->first;
        if (tmpFEDerror25.find(detid) != tmpFEDerror25.end()) {
          std::vector<PixelFEDChannel> chs = itFEDerror25->second;
          std::vector<PixelFEDChannel> chs_tmp = tmpFEDerror25[detid];

          std::vector<PixelFEDChannel> chs_common;
          for (unsigned int ich = 0; ich < chs.size(); ich++) {
            PixelFEDChannel ch = chs[ich];
            /* loop over the current FEDerror25 channels, save the common FED channels */
            for (unsigned int ich_tmp = 0; ich_tmp < chs_tmp.size(); ich_tmp++) {
              PixelFEDChannel ch_tmp = chs_tmp[ich_tmp];
              if ((ch.fed == ch_tmp.fed) && (ch.link == ch_tmp.link)) { /* the same FED channel */
                chs_common.push_back(ch);
                break;
              }
            }
          }
          /* remove the full module from FEDerror25 list if no common channels are left */
          if (chs_common.empty())
            fFEDerror25_.erase(itFEDerror25);
          /* otherwise replace with the common channels */
          else {
            fFEDerror25_[detid].clear();
            fFEDerror25_[detid] = chs_common;
          }
        } else { /* remove the full module from FEDerror25 list if the module doesn't appear in the current event's FEDerror25 list */
          fFEDerror25_.erase(itFEDerror25);
        }

      } /* loop over modules that have FEDerror25 in the first event in the lumi section */

    } /* non-empty FEDerror lists */

  } /* look over different resouces of takens */

  /* Caveat
     no per-event collection put into iEvent
     If use produce() function and no collection is put into iEvent, produce() will not run in unScheduled mode
     Now since CMSSW_10_1_X, the accumulate() function will run whatsoever in the unScheduled mode
     Accumulate() is NOT available for releases BEFORE CMSSW_10_1_X */
}

void SiPixelStatusProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
  /* set total number of events through ftotalevents_ */
  fDet_.setNevents(ftotalevents_);

  if (ftotalevents_ > 0) {
    /* Add FEDerror25 information into SiPixelDetectorStatus fDet_ for FED channels stored in fFEDerror25_ */
    if (!fFEDerror25_.empty()) {  // non-empty FEDerror25
      std::map<int, std::vector<PixelFEDChannel>>::iterator itFEDerror25;
      for (itFEDerror25 = fFEDerror25_.begin(); itFEDerror25 != fFEDerror25_.end();
           itFEDerror25++) {  // loop over detIds
        int detid = itFEDerror25->first;
        std::vector<PixelFEDChannel> chs = itFEDerror25->second;
        for (unsigned int ich = 0; ich < chs.size(); ich++) {
          fDet_.fillFEDerror25(detid, chs[ich]);
        }
      }  // loop over detIds
    }    // if non-empty FEDerror25

  }  // only for non-zero events
}

void SiPixelStatusProducer::endLuminosityBlockSummary(
    edm::LuminosityBlock const& iLumi,
    edm::EventSetup const&,
    std::vector<SiPixelDetectorStatus>* siPixelDetectorStatusVtr) const {
  /*add the Stream's partial information to the full information*/

  /* only save for the lumi sections with NON-ZERO events */
  if (ftotalevents_ > 0)
    siPixelDetectorStatusVtr->push_back(fDet_);
}

/* helper function */
int SiPixelStatusProducer::indexROC(int irow, int icol, int nROCcolumns) {
  return int(icol + irow * nROCcolumns);

  /* generate the folling roc index that is going to map with ROC id as
     8  9  10 11 12 13 14 15
     0  1  2  3  4  5  6  7 */
}

DEFINE_FWK_MODULE(SiPixelStatusProducer);
