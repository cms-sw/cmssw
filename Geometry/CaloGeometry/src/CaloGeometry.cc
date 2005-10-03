#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"


CaloGeometry::CaloGeometry() {
}

int CaloGeometry::makeIndex(DetId::Detector det, int subdet) const {
  return (int(det)<<4) | (subdet&0xF);
}

void CaloGeometry::setSubdetGeometry(DetId::Detector det, int subdet, const CaloSubdetectorGeometry* geom) {
  int index=makeIndex(det,subdet);
  theGeometries_[index]=geom;
}


const CaloSubdetectorGeometry* CaloGeometry::getSubdetectorGeometry(const DetId& id) const {
  std::map<int, const CaloSubdetectorGeometry*>::const_iterator i=theGeometries_.find(makeIndex(id.det(),id.subdetId()));
  return (i==theGeometries_.end())?(0):(i->second);
}

const CaloSubdetectorGeometry* CaloGeometry::getSubdetectorGeometry(DetId::Detector det, int subdet) const {
    std::map<int, const CaloSubdetectorGeometry*>::const_iterator i=theGeometries_.find(makeIndex(det,subdet));
    return (i==theGeometries_.end())?(0):(i->second);
}

static const GlobalPoint notFound(0,0,0);

const GlobalPoint& CaloGeometry::getPosition(const DetId& id) const {
    const CaloSubdetectorGeometry* geom=getSubdetectorGeometry(id);
    const CaloCellGeometry* cell=(geom==0)?(0):(geom->getGeometry(id));
    return (cell==0)?(notFound):(cell->getPosition());
}

const CaloCellGeometry* CaloGeometry::getGeometry(const DetId& id) const {
  const CaloSubdetectorGeometry* geom=getSubdetectorGeometry(id);
  const CaloCellGeometry* cell=(geom==0)?(0):(geom->getGeometry(id));
  return cell;
}

bool CaloGeometry::present(const DetId& id) const {
  const CaloSubdetectorGeometry* geom=getSubdetectorGeometry(id);
  return (geom==0)?(false):(geom->present(id));
  }

std::vector<DetId> CaloGeometry::getValidDetIds() const {
  std::vector<DetId> theList;
  std::map<int, const CaloSubdetectorGeometry*>::const_iterator i;
  for (i=theGeometries_.begin(); i!=theGeometries_.end(); i++) {
    DetId::Detector det=(DetId::Detector)(i->first>>4);
    int subdet=i->first&0xF;
    std::vector<DetId> aList=i->second->getValidDetIds(det,subdet);
    theList.insert(theList.end(),aList.begin(),aList.end());
  }
  return theList;
}

std::vector<DetId> CaloGeometry::getValidDetIds(DetId::Detector det, int subdet) const {
  std::vector<DetId> theList;
  std::map<int, const CaloSubdetectorGeometry*>::const_iterator i=theGeometries_.find(makeIndex(det,subdet));
  if (i!=theGeometries_.end()) {
    theList=i->second->getValidDetIds(det,subdet);
  }
  return theList;
}
  

