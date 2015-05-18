#ifndef CALIBFORMATS_HCALOBJECTS_HCALCALIBRATIONWIDTHSSET_H
#define CALIBFORMATS_HCALOBJECTS_HCALCALIBRATIONWIDTHSSET_H 1

#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include "CondFormats/HcalObjects/interface/HcalDetIdRelationship.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include <vector>

/** \class HcalCalibrationWidthsSet
  *  
  * \author R.Ofierzynski
  */
class HcalCalibrationWidthsSet {
public:
  HcalCalibrationWidthsSet();
  const HcalCalibrationWidths& getCalibrationWidths(const DetId id) const;
  void setCalibrationWidths(const DetId id, const HcalCalibrationWidths& ca);
  void sort();
  void clear();
private:
  struct CalibWidthSetObject {
    CalibWidthSetObject(const DetId& aid) {
      id = hcalTransformedId(aid);
    }
    DetId id;
    HcalCalibrationWidths calib;
    bool operator<(const CalibWidthSetObject& cso) const { return id < cso.id; }
    bool operator==(const CalibWidthSetObject& cso) const { return id == cso.id; }
  };
  typedef CalibWidthSetObject Item;
  HcalCalibrationWidths dummy;
  std::vector<CalibWidthSetObject> mItems;
  bool sorted_;
};

#endif
