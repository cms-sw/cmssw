#ifndef CALIBFORMATS_HCALOBJECTS_HCALCALIBRATIONSSET_H
#define CALIBFORMATS_HCALOBJECTS_HCALCALIBRATIONSSET_H 1

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CondFormats/HcalObjects/interface/HcalDetIdRelationship.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include <vector>

/** \class HcalCalibrationsSet
  *  
  * \author J. Mans - Minnesota
  */
class HcalCalibrationsSet {
public:
  HcalCalibrationsSet();
  const HcalCalibrations& getCalibrations(const DetId id) const;
  void setCalibrations(const DetId id, const HcalCalibrations& ca);
  void sort();
  void clear();
private:
  struct CalibSetObject {
    CalibSetObject(const DetId& aid) {
      id = hcalTransformedId(aid);
    }
    DetId id;
    HcalCalibrations calib;
    bool operator<(const CalibSetObject& cso) const { return id < cso.id; }
    bool operator==(const CalibSetObject& cso) const { return id == cso.id; }
  };
  typedef CalibSetObject Item;
  HcalCalibrations dummy;
  std::vector<CalibSetObject> mItems;
  bool sorted_;
};

#endif
