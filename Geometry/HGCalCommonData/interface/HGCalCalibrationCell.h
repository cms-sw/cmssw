#ifndef Geometry_HGCalCommonData_HGCalCalibrationCell_h
#define Geometry_HGCalCommonData_HGCalCalibrationCell_h

#include "Geometry/HGCalCommonData/interface/HGCalCell.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include <memory>
#include <vector>

class HGCalCalibrationCell {
public:
  HGCalCalibrationCell(const HGCalDDDConstants* cons);
  HGCalCalibrationCell() {}

  int findCell(int zside, int layer, int waferU, int waferV, int cellUV, const std::pair<double, double>& xy) const;

private:
  const HGCalDDDConstants* cons_;
  std::unique_ptr<HGCalCell> wafer_;
  double radius_[2];
  std::vector<int> cells_[4];
};

#endif
