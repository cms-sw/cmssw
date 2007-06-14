#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"


CaloSubdetectorGeometry::~CaloSubdetectorGeometry() { 
  CellCont::iterator i=cellGeometries_.begin();
  for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++)
    delete const_cast<CaloCellGeometry*>((*i).second);
}

void CaloSubdetectorGeometry::addCell(const DetId& id, const CaloCellGeometry* ccg) {
  cellGeometries_.insert(std::make_pair(id,ccg));
}

const CaloCellGeometry* CaloSubdetectorGeometry::getGeometry(const DetId& id) const {
  CellCont::const_iterator i=cellGeometries_.find(id);
  return i==(cellGeometries_.end())?(0):(i->second);
}

bool CaloSubdetectorGeometry::present(const DetId& id) const {
  CellCont::const_iterator i=cellGeometries_.find(id);
  return i!=cellGeometries_.end();
}


std::vector<DetId> const & CaloSubdetectorGeometry::getValidDetIds(DetId::Detector det, int subdet) const {
  if (validIds_.empty()) {
    validIds_.reserve(cellGeometries_.size());
    CellCont::const_iterator i;
    for (i=cellGeometries().begin(); i!=cellGeometries().end(); i++)
      validIds_.push_back(i->first);
    std::sort(validIds_.begin(),validIds_.end());
  }

  return validIds_;    
}

double CaloSubdetectorGeometry::deltaR(const GlobalPoint& p1, const GlobalPoint& p2) {
  double dp=p1.phi()-p2.phi();
  double de=p1.eta()-p2.eta();
  return std::sqrt(dp*dp+de*de);
}

DetId CaloSubdetectorGeometry::getClosestCell(const GlobalPoint& r) const 
{
  CellCont::const_iterator i;
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

  CellCont::const_iterator i;
  for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++) {
    double dist=deltaR(r,i->second->getPosition());
    if (dist<=dR) dss.insert(i->first);
  }   

  return dss;
}
