#include "CalibTracker/SiPixelTools/interface/SiPixelFrameReverter.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

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

SiPixelFrameReverter::SiPixelFrameReverter(const edm::EventSetup& iSetup, const SiPixelFedCablingTree * tree) 
  : tree_(tree)
{ 
  // Build tree
  buildStructure(iSetup);
}


void SiPixelFrameReverter::buildStructure(const edm::EventSetup& iSetup) 
{

  // First build SiPixelFrameConverter for each FED

  int fedId;

  std::map<int,SiPixelFrameConverter*> FedToConverterMap;

  for(fedId = 0; fedId <= 39; fedId++){
    
    SiPixelFrameConverter * converter = (tree_ ) ? 
      new SiPixelFrameConverter(tree_, fedId) : 0;

    FedToConverterMap.insert(pair<int,SiPixelFrameConverter*> (fedId,converter));

  }

  // Next create map connecting each detId to appropriate SiPixelFrameConverter

  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );

  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    
    if(dynamic_cast<PixelGeomDetUnit*>((*it))!=0){

      DetId detId = (*it)->geographicalId();
      const GeomDetUnit      * geoUnit = pDD->idToDetUnit( detId );

      if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
	uint32_t id = detId();

	for (fedId = 0; fedId <= 31; fedId++) {
	  if ( !(*tree_).fed(fedId) ) continue;
	  const PixelFEDCabling & theFed = *(*tree_).fed(fedId);
	  for (int idxLink = 1; idxLink <= theFed.numberOfLinks(); idxLink++) {
	    const PixelFEDLink * link = theFed.link(idxLink);
	    if (!link) continue;
	    int numberOfRocs = link->numberOfROCs();
	    for(int idxRoc = 1; idxRoc <= numberOfRocs; idxRoc++) {
	      const PixelROC * roc = link->roc(idxRoc);
	      if (!roc) continue;
	      if (id == roc->rawId() ) {
		FEDType fed;
		fed.first = fedId;
		fed.second = FedToConverterMap[fedId];
		DetToFedMap.insert(pair< uint32_t,FEDType > (id,fed));
	      }  // if (rawId
	    }  // for(int idxRoc
	  }  // for (int idxLink
	}  // for (fedId

      }	else if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
	uint32_t id = detId();

	for (fedId = 32; fedId <= 39; fedId++) {
	  if ( !(*tree_).fed(fedId) ) continue;
	  const PixelFEDCabling & theFed = *(*tree_).fed(fedId);
	  for (int idxLink = 1; idxLink <= theFed.numberOfLinks(); idxLink++) {
	    const PixelFEDLink * link = theFed.link(idxLink);
	    if (!link) continue;
	    int numberOfRocs = link->numberOfROCs();
	    for(int idxRoc = 1; idxRoc <= numberOfRocs; idxRoc++) {
	      const PixelROC * roc = link->roc(idxRoc);
	      if (!roc) continue;
	      if (id == roc->rawId() ) {
		FEDType fed;
		fed.first = fedId;
		fed.second = FedToConverterMap[fedId];
		DetToFedMap.insert(pair< uint32_t,FEDType > (id,fed));
	      }  // if (rawId
	    }  // for(int idxRoc
	  }  // for (int idxLink 
	}  // for (fedId
      }  // else if(detId.subdetId()
    }  // if(dynamic_cast<PixelGeomDetUnit*>
  }  // for(TrackerGeometry::DetContainer::const_iterator

}  // end buildStructure


int SiPixelFrameReverter::findFedId(uint32_t detId)
{
  return DetToFedMap[detId].first;
}


int SiPixelFrameReverter::findLinkInFed(uint32_t detId, int row, int col)
{
  DetectorIndex detector = {detId, row, col};
  ElectronicIndex  cabling;
  int status  = DetToFedMap[detId].second->toCabling(cabling, detector);
  if (status) {
    edm::LogError("SiPixelFrameReverter::findLinkInFed") << " Error: status "<<status<<" returned";
    return -1;
  } else return cabling.link;
}


int SiPixelFrameReverter::findRocInLink(uint32_t detId, int row, int col)
{
  DetectorIndex detector = {detId, row, col};
  ElectronicIndex  cabling;
  int status  = DetToFedMap[detId].second->toCabling(cabling, detector);
  if (status) {
    edm::LogError("SiPixelFrameReverter::findRocInLink") << " Error: status "<<status<<" returned";
    return -1;
  } else return cabling.roc;
}


int SiPixelFrameReverter::findRocInDet(uint32_t detId, int row, int col)
{
  DetectorIndex detector = {detId, row, col};
  ElectronicIndex  cabling;
  int status  = DetToFedMap[detId].second->toCabling(cabling, detector);
  if (status) {
    edm::LogError("SiPixelFrameReverter::findRocInDet") << " Error: status "<<status<<" returned";
    return -1;
  } else {
    int fedId = DetToFedMap[detId].first;
    const sipixelobjects::PixelFEDCabling & theFed = *(*tree_).fed( fedId );
    const PixelFEDLink * link = theFed.link( cabling.link);
    const PixelROC * roc = link->roc(cabling.roc);
    int rocInDet = roc->idInDetUnit();
    return rocInDet;
  }
}


LocalPixel SiPixelFrameReverter::findPixelInRoc(uint32_t detId, int row, int col)
{
  DetectorIndex detector = {detId, row, col};
  ElectronicIndex  cabling;
  int status  = DetToFedMap[detId].second->toCabling(cabling, detector);
  if (status) {
    edm::LogError("SiPixelFrameReverter::findPixelInROC") << " Error: status "<<status<<" returned";
  } else {
    int fedId = DetToFedMap[detId].first;
    const sipixelobjects::PixelFEDCabling & theFed = *(*tree_).fed( fedId );
    const PixelFEDLink * link = theFed.link( cabling.link);
    const PixelROC * roc = link->roc(cabling.roc);
    GlobalPixel global = {row, col};
    LocalPixel  local = roc->toLocal(global);
    return local;
  }
}

/*
LocalPixel::DcolPxid SiPixelFrameReverter::findPixelInDcol(uint32_t detId, int row, int col)
{
  DetectorIndex detector = {detId, row, col};
  ElectronicIndex  cabling;
  int status  = DetToFedMap[detId].second->toCabling(cabling, detector);
  if (status) {
    edm::LogError("SiPixelFrameReverter::findPixelInDcol") << " Error: status "<<status<<" returned";
    return -1;
  } else {
    LocalPixel::DcolPxid local = { cabling.dcol, cabling.pxid };
    return local;
  }
}
*/
