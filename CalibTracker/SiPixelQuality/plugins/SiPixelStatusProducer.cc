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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESWatcher.h"

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
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

// EDProducer related dataformat
#include "DQM/SiPixelPhase1Common/interface/SiPixelCoordinates.h"
#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"

// header file
#include "CalibTracker/SiPixelQuality/plugins/SiPixelStatusProducer.h"

using namespace std;

//--------------------------------------------------------------------------------------------------
SiPixelStatusProducer::SiPixelStatusProducer(const edm::ParameterSet& iConfig){
  // get parameter

  // badPixelFEDChannelCollections
  std::vector<edm::InputTag> badPixelFEDChannelCollectionLabels_ = iConfig.getParameter<edm::ParameterSet>("SiPixelStatusProducerParameters").getParameter<std::vector<edm::InputTag> >("badPixelFEDChannelCollections");
  for (auto &t : badPixelFEDChannelCollectionLabels_)
      theBadPixelFEDChannelsTokens_.push_back(consumes<PixelFEDChannelCollection>(t));
  //  badPixelFEDChannelCollections = cms.VInputTag(cms.InputTag('siPixelDigis'))

  fPixelClusterLabel_   = iConfig.getParameter<edm::ParameterSet>("SiPixelStatusProducerParameters").getUntrackedParameter<edm::InputTag>("pixelClusterLabel");
  fSiPixelClusterToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(fPixelClusterLabel_);
  resetNLumi_   = iConfig.getParameter<edm::ParameterSet>("SiPixelStatusProducerParameters").getUntrackedParameter<int>("resetEveryNLumi",1);

  ftotalevents = 0;
  countLumi_ = 0;

  beginLumi_ = endLumi_ = -1;
  endLumi_ = endRun_ = -1;
  
  produces<SiPixelDetectorStatus, edm::Transition::EndLuminosityBlock>("siPixelStatus");
}

//--------------------------------------------------------------------------------------------------
SiPixelStatusProducer::~SiPixelStatusProducer(){

}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup){

  edm::LogInfo("SiPixelStatusProducer")
                 << "beginlumi setup "<<endl;

  if ( countLumi_ == 0 && resetNLumi_ > 0 ) {

    beginLumi_ = lumiSeg.luminosityBlock();
    beginRun_ = lumiSeg.run();
    ftotalevents = 0;

  }

  // The es watcher is acutally not needed if run parallel jobs for each lumi section
  if(siPixelFedCablingMapWatcher_.check(iSetup)||trackerDIGIGeoWatcher_.check(iSetup)||trackerTopoWatcher_.check(iSetup)){

    coord_.init(iSetup);
    iSetup.get<SiPixelFedCablingMapRcd>().get(fCablingMap);
    fCablingMap_ = fCablingMap.product();
    iSetup.get<TrackerDigiGeometryRecord>().get(fTG);

    fFedIds = fCablingMap_->det2fedMap();

  } // if conditionWatcher_.check(iSetup)

  // init the SiPixelDetectorStatus fDet and sensor size fSensors in the begining (when countLumi is zero)
  if(countLumi_ == 0){

     for (TrackerGeometry::DetContainer::const_iterator it = fTG->dets().begin(); it != fTG->dets().end(); it++){

          const PixelGeomDetUnit *pgdu = dynamic_cast<const PixelGeomDetUnit*>((*it));
          if (pgdu == nullptr) continue;
          DetId detId = (*it)->geographicalId();
          int detid = detId.rawId();

          // don't want to use magic number row 80 column 52
          const PixelTopology* topo = static_cast<const PixelTopology*>(&pgdu->specificTopology());
          int rowsperroc = topo->rowsperroc();
          int colsperroc = topo->colsperroc();

          int nROCrows = pgdu->specificTopology().nrows()/rowsperroc;
          int nROCcolumns = pgdu->specificTopology().ncolumns()/colsperroc;
          int nrocs = nROCrows*nROCcolumns;

          fDet.addModule(detid, nrocs);

          fSensors[detid] = std::make_pair(rowsperroc,colsperroc);
          fSensorLayout[detid] = std::make_pair(nROCrows,nROCcolumns);

          std::map<int, int> rocIdMap;
          for(int irow = 0; irow<nROCrows; irow++){

             for(int icol = 0; icol<nROCcolumns; icol++){

                 int dummyOfflineRow = (rowsperroc/2-1) + irow*rowsperroc;
                 int dummeOfflineColumn = (colsperroc/2-1) + icol*colsperroc;
                 // encode irow, icol
                 int key = indexROC(irow,icol,nROCcolumns);

                 int roc(-1), rocR(-1), rocC(-1);
                 onlineRocColRow(detId, dummyOfflineRow, dummeOfflineColumn, roc, rocR, rocC);

                 int value = roc;
                 rocIdMap[key] = value;
             }

          }

         fRocIds[detid] = rocIdMap;
     }

  } // init when countLumi = 0

  countLumi_++;

}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusProducer::accumulate(edm::Event const& iEvent, const edm::EventSetup& iSetup){

  ftotalevents++;

  edm::LogInfo("SiPixelStatusProducer")
                 << "start cluster analyzer "<<endl;

  // ----------------------------------------------------------------------
  // -- Pixel cluster analysis
  // ----------------------------------------------------------------------

  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > hClusterColl;
  if(!iEvent.getByToken(fSiPixelClusterToken_, hClusterColl)){
     edm::LogWarning("SiPixelStatusProducer") << " edmNew::DetSetVector<SiPixelCluster> "<<fPixelClusterLabel_<<" does not exist!"<<endl;
     return;
  }

  iEvent.getByToken(fSiPixelClusterToken_, hClusterColl);

  if (hClusterColl.isValid()){

     for (const auto& clusters: *hClusterColl) { //loop over different clusters in a clusters vector (module)
           for(const auto& clu: clusters) { // loop over cluster in a given detId (module)
              int detid = clusters.detId();
              int rowsperroc = fSensors[detid].first;
              int colsperroc = fSensors[detid].second;

              int nROCcolumns = fSensorLayout[detid].second;

              int roc(-1);
              std::map<int,int> fRocIds_detid;
              if(fRocIds.find(detid)!=fRocIds.end()){
                fRocIds_detid = fRocIds[detid];
              }

              const vector<SiPixelCluster::Pixel>& pixvector = clu.pixels();
              for (unsigned int i = 0; i < pixvector.size(); ++i) {

                   int mr0 = pixvector[i].x; // constant column direction is along x-axis,
                   int mc0 = pixvector[i].y; // constant row direction is along y-axis

                   int irow = mr0/rowsperroc;
                   int icol = mc0/colsperroc;

                   int key = indexROC(irow,icol,nROCcolumns);
                   if(fRocIds_detid.find(key)!=fRocIds_detid.end()){
                      roc = fRocIds_detid[key];
                   }

                   fDet.fillDIGI(detid, roc);

              }// loop over pixels in a given cluster

           }// loop over cluster in a given detId (module)

    }// loop over detId-grouped clusters in cluster detId-grouped clusters-vector

  } // hClusterColl.isValid()
  else{
      edm::LogWarning("SiPixelStatusProducer") << " edmNew::DetSetVector<SiPixelCluster> "<<fPixelClusterLabel_<<" is NOT Valid!"<<endl;
  }
  //////////////////////////////////////////////////////////////////////

  // FEDerror25 per-ROC per-event
  edm::Handle<PixelFEDChannelCollection> pixelFEDChannelCollectionHandle;

  // look over different resouces of takens
  for (const edm::EDGetTokenT<PixelFEDChannelCollection>& tk: theBadPixelFEDChannelsTokens_) {

        // collection has to exist
        if (!iEvent.getByToken(tk, pixelFEDChannelCollectionHandle)) {
           edm::LogWarning("SiPixelStatusProducer") << " PixelFEDChannelCollection with index "<<tk.index()<<" does NOT exist!"<<endl;
           continue;
        }
        iEvent.getByToken(tk, pixelFEDChannelCollectionHandle);
        // collection has to be valid
        if(!pixelFEDChannelCollectionHandle.isValid()) {
           edm::LogWarning("SiPixelStatusProducer") << " PixelFEDChannelCollection with index "<<tk.index()<<" is NOT valid!"<<endl;
           continue;
        }
        for (const auto& disabledChannels: *pixelFEDChannelCollectionHandle) {
            //loop over different PixelFED in a PixelFED vector (module)
            for(const auto& ch: disabledChannels) {

                DetId detId = disabledChannels.detId();
                int detid = detId.rawId();
                fDet.fillFEDerror25(detid,ch);

            } // loop over different PixelFED in a PixelFED vector (different channel for a given module)

        }// loop over different (different DetId) PixelFED vectors in PixelFEDChannelCollection

   }   // look over different resouces of takens

   // no per-event collection put into iEvent 
   // If use produce() function and no collection is put into iEvent, produce() will not run in unScheduled mode
   // Now since CMSSW_10_1_X, the accumulate() function will run whatsoever in the unScheduled mode
   // But accumulate() is NOT available and will NOT be available for releases before CMSSW_10_1_X

}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup){

}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusProducer::endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup){

  edm::LogInfo("SiPixelStatusProducer")
                 << "endlumi producer "<<endl;

  endLumi_ = lumiSeg.luminosityBlock();
  endRun_  = lumiSeg.run();

  // check if countLumi_ is large enough to read out/save data and reset for the next round
  if ( resetNLumi_ == -1 ) return;
  if ( countLumi_ < resetNLumi_ ) return;

  fDet.setRunRange(beginRun_,endRun_);
  fDet.setLSRange(beginLumi_,endLumi_);
  fDet.setNevents(ftotalevents);

  // save result
  auto result = std::make_unique<SiPixelDetectorStatus>();
  *result = fDet;

  // only save for the lumi sections with NON-ZERO events
  lumiSeg.put(std::move(result), std::string("siPixelStatus"));
  edm::LogInfo("SiPixelStatusProducer")
                 << "new lumi-based data stored for run "<<beginRun_<<" lumi from "<<beginLumi_<<" to "<<endLumi_<<std::endl;

  // reset detector status and lumi-counter
  fDet.resetDetectorStatus();
  countLumi_=0;
  ftotalevents=0;

}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusProducer::onlineRocColRow(const DetId &detId, int offlineRow, int offlineCol, int &roc, int &row, int &col) {

  int fedId = fFedIds[detId.rawId()];

  // from detector to cabling
  sipixelobjects::ElectronicIndex cabling;
  sipixelobjects::DetectorIndex detector; //{detId.rawId(), offlineRow, offlineCol};
  detector.rawId = detId.rawId(); detector.row = offlineRow; detector.col = offlineCol;

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
  //const sipixelobjects::PixelROC *theRoc = fCablingMap->findItem(path);
  const sipixelobjects::PixelROC *theRoc = converter.toRoc(cabling.link, cabling.roc);
  roc = theRoc->idInDetUnit();

  // has to be BPIX; has to be minus side; has to be half module
  // for phase-I, there is no half module
  if (detId.subdetId() == PixelSubdetector::PixelBarrel && coord_.side(detId)==1 && coord_.half(detId))
  {
      roc += 8;
  }

}

//--------------------------------------------------------------------------------------------------
int SiPixelStatusProducer::indexROC(int irow, int icol, int nROCcolumns){

    return int(icol+irow*nROCcolumns);

    // generate the folling roc index that is going to map with ROC id as
    // 8  9  10 11 12 13 14 15
    // 0  1  2  3  4  5  6  7

}

DEFINE_FWK_MODULE(SiPixelStatusProducer);
