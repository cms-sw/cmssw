#include "FastSimulation/CaloGeometryTools/interface/DistanceToCell.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

DistanceToCell::DistanceToCell() : det_(nullptr) { ; }

DistanceToCell::DistanceToCell(const DistanceToCell& dist) {
  det_ = dist.det_;
  pivotPosition_ = dist.pivotPosition_;
  pivot_ = dist.pivot_;
}

DistanceToCell::DistanceToCell(const CaloSubdetectorGeometry* det, const DetId& cell) : det_(det), pivot_(cell) {
  pivotPosition_ = (cell.det() == DetId::Hcal) ? (static_cast<const HcalGeometry*>(det_))->getPosition(cell)
                                               : det_->getGeometry(pivot_)->getPosition();
}

bool DistanceToCell::operator()(const DetId& c1, const DetId& c2) {
  bool ok = (c1.det() == DetId::Hcal)
                ? (((static_cast<const HcalGeometry*>(det_))->getPosition(c1) - pivotPosition_).mag2() <
                   ((static_cast<const HcalGeometry*>(det_))->getPosition(c2) - pivotPosition_).mag2())
                : ((det_->getGeometry(c1)->getPosition() - pivotPosition_).mag2() <
                   (det_->getGeometry(c2)->getPosition() - pivotPosition_).mag2());
  return ok;
}
