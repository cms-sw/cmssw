#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
// DataFormats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

using namespace std;
using namespace sipixelobjects;

SiPixelFrameReverter::SiPixelFrameReverter(const edm::EventSetup& iSetup, const SiPixelFedCabling* map) 
  : map_(map)
{ 
  // Build map
  buildStructure(iSetup);
}


void SiPixelFrameReverter::buildStructure(const edm::EventSetup& iSetup) 
{

  // Create map connecting each detId to appropriate SiPixelFrameConverter

  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );

  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    
    if(dynamic_cast<PixelGeomDetUnit*>((*it))!=0){

      DetId detId = (*it)->geographicalId();
      uint32_t id = detId();
      std::vector<CablingPathToDetUnit> paths = map_->pathToDetUnit(id);
      DetToFedMap.insert(pair< uint32_t,std::vector<CablingPathToDetUnit> > (id,paths));

    }
  }  // for(TrackerGeometry::DetContainer::const_iterator
}  // end buildStructure


int SiPixelFrameReverter::toCabling(
    sipixelobjects::ElectronicIndex & cabling, const sipixelobjects::DetectorIndex & detector) const
{
  if(!hasDetUnit(detector.rawId)) return -1;
  std::vector<CablingPathToDetUnit> path = DetToFedMap.find(detector.rawId)->second;
  typedef std::vector<CablingPathToDetUnit>::const_iterator IT;
  for  (IT it = path.begin(); it != path.end(); ++it) {
    const PixelROC * roc = map_->findItem(*it);
    if (!roc) return -3;
    if (! roc->rawId() == detector.rawId) return -4;

    GlobalPixel global = {detector.row, detector.col};
    LocalPixel local = roc->toLocal(global);
    if(!local.valid()) continue;
    ElectronicIndex cabIdx = {static_cast<int>(it->link), static_cast<int>(it->roc), local.dcol(), local.pxid()};
    cabling = cabIdx;

    return it->fed;
  }
  return -2;
}


int SiPixelFrameReverter::findFedId(uint32_t detId)
{
  if(!hasDetUnit(detId)) return -1;
  std::vector<CablingPathToDetUnit> path = DetToFedMap.find(detId)->second;
  int fedId = (int) path[0].fed;
  return fedId;
}


short SiPixelFrameReverter::findLinkInFed(uint32_t detId, GlobalPixel global)
{
  if(!hasDetUnit(detId)) return -1;
  std::vector<CablingPathToDetUnit> path = DetToFedMap.find(detId)->second;
  typedef  std::vector<CablingPathToDetUnit>::const_iterator IT;
  for  (IT it = path.begin(); it != path.end(); ++it) {
    const PixelROC * roc = map_->findItem(*it);
    if (!roc) continue;

    LocalPixel local = roc->toLocal(global);

    if(!local.valid()) continue;
    short link = (short) it->link;
    return link;
  }
  return -1;
}


short SiPixelFrameReverter::findRocInLink(uint32_t detId, GlobalPixel global)
{
  if(!hasDetUnit(detId)) return -1;
  std::vector<CablingPathToDetUnit> path = DetToFedMap.find(detId)->second;
  typedef  std::vector<CablingPathToDetUnit>::const_iterator IT;
  for  (IT it = path.begin(); it != path.end(); ++it) {
    const PixelROC * roc = map_->findItem(*it);
    if (!roc) continue;

    LocalPixel local = roc->toLocal(global);

    if(!local.valid()) continue;
    short rocInLink = (short) roc->idInLink();
    return rocInLink;
  }
  return -1;
}


short SiPixelFrameReverter::findRocInDet(uint32_t detId, GlobalPixel global)
{
  if(!hasDetUnit(detId)) return -1;
  std::vector<CablingPathToDetUnit> path = DetToFedMap.find(detId)->second;
  typedef  std::vector<CablingPathToDetUnit>::const_iterator IT;
  for  (IT it = path.begin(); it != path.end(); ++it) {
    const PixelROC * roc = map_->findItem(*it);
    if (!roc) continue;

    LocalPixel local = roc->toLocal(global);

    if(!local.valid()) continue;
    short rocInDet = (short) roc->idInDetUnit();
    return rocInDet;
  }
  return -1;
}


LocalPixel SiPixelFrameReverter::findPixelInRoc(uint32_t detId, GlobalPixel global)
{
  if(!hasDetUnit(detId)) {
    LocalPixel::RocRowCol pixel = {-1,-1};
    LocalPixel local(pixel);
    return local;
  }
  std::vector<CablingPathToDetUnit> path = DetToFedMap.find(detId)->second;
  typedef  std::vector<CablingPathToDetUnit>::const_iterator IT;
  for  (IT it = path.begin(); it != path.end(); ++it) {
    const PixelROC * roc = map_->findItem(*it);
    if (!roc) continue;

    LocalPixel local = roc->toLocal(global);

    if(!local.valid()) continue;
    return local;
  }
  LocalPixel::RocRowCol pixel = {-1,-1};
  LocalPixel local(pixel);
  return local;
}
