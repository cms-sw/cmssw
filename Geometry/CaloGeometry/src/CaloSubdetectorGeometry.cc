#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"


void CaloSubdetectorGeometry::addCell(const DetId& id, const CaloCellGeometry* ccg) {
  cellGeometries_.insert(std::pair<DetId,const CaloCellGeometry*>(id,ccg));
}

const CaloCellGeometry* CaloSubdetectorGeometry::getGeometry(const DetId& id) const {
  std::map<DetId, const CaloCellGeometry*>::const_iterator i=cellGeometries_.find(id);
  return i==(cellGeometries_.end())?(0):(i->second);
}

bool CaloSubdetectorGeometry::present(const DetId& id) const {
  std::map<DetId, const CaloCellGeometry*>::const_iterator i=cellGeometries_.find(id);
  return i!=cellGeometries_.end();
}


std::vector<DetId> CaloSubdetectorGeometry::getValidDetIds(DetId::Detector det, int subdet) const {
  if (validIds_.empty()) {
    std::map<DetId, const CaloCellGeometry*>::const_iterator i;
    for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++)
      validIds_.push_back(i->first);
  }

  return validIds_;    
}

double CaloSubdetectorGeometry::deltaR(const GlobalPoint& p1, const GlobalPoint& p2) {
  double dp=p1.phi()-p2.phi();
  double de=p1.eta()-p2.eta();
  return sqrt(dp*dp+de*de);
}

const DetId CaloSubdetectorGeometry::getClosestCell(const GlobalPoint& r) const 
{
  std::map<DetId, const CaloCellGeometry*>::const_iterator i;
  double closest=1e5;
  DetId retval(0);
  for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++) {
    double dR=deltaR(r,i->second->getPosition());
    if (dR<closest) {
      closest=dR;
      retval=i->first;
    }
  }   

  return retval;
}


CaloSubdetectorGeometry::DetIdSet CaloSubdetectorGeometry::getCells(const GlobalPoint& r, double dR) const {
  DetIdSet dss;

  std::map<DetId, const CaloCellGeometry*>::const_iterator i;
  for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++) {
    double dist=deltaR(r,i->second->getPosition());
    if (dist<=dR) dss.insert(i->first);
  }   

  return dss;
}
