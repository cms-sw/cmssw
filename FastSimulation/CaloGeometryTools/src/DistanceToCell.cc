#include "FastSimulation/CaloGeometryTools/interface/DistanceToCell.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

DistanceToCell::DistanceToCell():det_(0) {;}

DistanceToCell::DistanceToCell(const DistanceToCell& dist)
{
  det_= dist.det_;
  pivotPosition_ = dist.pivotPosition_;
  pivot_= dist.pivot_;
}

DistanceToCell::DistanceToCell(const CaloSubdetectorGeometry * det,const  DetId& cell):det_(det),pivot_(cell)
{
  pivotPosition_ = det_->getGeometry(pivot_)->getPosition();
}

bool DistanceToCell::operator() (const DetId & c1, const DetId & c2)
{
  return ((det_->getGeometry(c1)->getPosition()-pivotPosition_).mag2()<
	  (det_->getGeometry(c2)->getPosition()-pivotPosition_).mag2());
}
