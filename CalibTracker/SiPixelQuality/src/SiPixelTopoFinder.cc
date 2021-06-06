
#include "CalibTracker/SiPixelQuality/interface/SiPixelTopoFinder.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

SiPixelTopoFinder::SiPixelTopoFinder() {}

SiPixelTopoFinder::~SiPixelTopoFinder() {}

void SiPixelTopoFinder::init(const TrackerGeometry* trackerGeometry,
                             const TrackerTopology* trackerTopology,
                             const SiPixelFedCablingMap* siPixelFedCablingMap) {
  phase_ = -1;

  tkGeom_ = trackerGeometry;
  tkTopo_ = trackerTopology;
  cablingMap_ = siPixelFedCablingMap;
  // from cabling map to FedIds
  fFedIds_ = cablingMap_->det2fedMap();

  // clear data
  fDetIds_.clear();
  fSensors_.clear();
  fSensorLayout_.clear();
  fRocIds_.clear();

  // loop over tracker geometry
  for (TrackerGeometry::DetContainer::const_iterator it = tkGeom_->dets().begin(); it != tkGeom_->dets().end();
       it++) {  // tracker geo

    const PixelGeomDetUnit* pgdu = dynamic_cast<const PixelGeomDetUnit*>((*it));
    if (pgdu == nullptr)
      continue;
    // get detId for a module
    DetId detId = (*it)->geographicalId();
    int detid = detId.rawId();
    fDetIds_.push_back(detid);

    // don't want to use magic number row 80 column 52 for Phase-1
    const PixelTopology* topo = static_cast<const PixelTopology*>(&pgdu->specificTopology());
    int rowsperroc = topo->rowsperroc();
    int colsperroc = topo->colsperroc();

    int nROCrows = pgdu->specificTopology().nrows() / rowsperroc;
    int nROCcolumns = pgdu->specificTopology().ncolumns() / colsperroc;

    fSensors_[detid] = std::make_pair(rowsperroc, colsperroc);
    fSensorLayout_[detid] = std::make_pair(nROCrows, nROCcolumns);

    std::map<int, int> rocIdMap;
    for (int irow = 0; irow < nROCrows; irow++) {       // row
      for (int icol = 0; icol < nROCcolumns; icol++) {  // column
        int dummyOfflineRow = (rowsperroc / 2 - 1) + irow * rowsperroc;
        int dummeOfflineColumn = (colsperroc / 2 - 1) + icol * colsperroc;
        int fedId = fFedIds_[detId.rawId()];

        int roc(-1), rocR(-1), rocC(-1);
        SiPixelTopoFinder::onlineRocColRow(
            detId, cablingMap_, fedId, dummyOfflineRow, dummeOfflineColumn, roc, rocR, rocC);

        // encode irow, icol
        int key = SiPixelTopoFinder::indexROC(irow, icol, nROCcolumns);
        int value = roc;
        rocIdMap[key] = value;
      }  // column
    }    // row

    fRocIds_[detid] = rocIdMap;

  }  // tracker geo
}

void SiPixelTopoFinder::onlineRocColRow(const DetId& detId,
                                        const SiPixelFedCablingMap* cablingMap,
                                        int fedId,
                                        int offlineRow,
                                        int offlineCol,
                                        int& roc,
                                        int& row,
                                        int& col) {
  // from detector to cabling
  sipixelobjects::ElectronicIndex cabling;
  sipixelobjects::DetectorIndex detector;  //{detId.rawId(), offlineRow, offlineCol};
  detector.rawId = detId.rawId();
  detector.row = offlineRow;
  detector.col = offlineCol;

  SiPixelFrameConverter converter(cablingMap, fedId);
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
  if (detId.subdetId() == PixelSubdetector::PixelBarrel && SiPixelTopoFinder::side(detId) == 1 &&
      SiPixelTopoFinder::half(detId)) {
    roc += 8;
  }
}

int SiPixelTopoFinder::indexROC(int irow, int icol, int nROCcolumns) {
  return int(icol + irow * nROCcolumns);

  /*generate the folling roc index that is going to map with ROC id as
     8  9  10 11 12 13 14 15
     0  1  2  3  4  5  6  7  */
}

// The following three functions copied from DQM/SiPixelPhase1Common/src/SiPixelCoordinates.cc
int SiPixelTopoFinder::quadrant(const DetId& detid) {
  if (detid.subdetId() == PixelSubdetector::PixelBarrel)
    return PixelBarrelName(detid, tkTopo_, phase_).shell();
  else if (detid.subdetId() == PixelSubdetector::PixelEndcap)
    return PixelEndcapName(detid, tkTopo_, phase_).halfCylinder();
  else
    return -9999;
}

int SiPixelTopoFinder::side(const DetId& detid) {
  if (detid.subdetId() == PixelSubdetector::PixelBarrel)
    return 1 + (SiPixelTopoFinder::quadrant(detid) > 2);
  else if (detid.subdetId() == PixelSubdetector::PixelEndcap)
    return tkTopo_->pxfSide(detid);
  else
    return -9999;
}

int SiPixelTopoFinder::half(const DetId& detid) {
  if (detid.subdetId() == PixelSubdetector::PixelBarrel)
    return PixelBarrelName(detid, tkTopo_, phase_).isHalfModule();
  else
    return -9999;
}
