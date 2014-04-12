#ifndef CALIBFORMATS_CASTOROBJECTS_CASTORCALIBRATIONSSET_H
#define CALIBFORMATS_CASTOROBJECTS_CASTORCALIBRATIONSSET_H 1

#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <vector>

class CastorCalibrationsSet {
public:
  CastorCalibrationsSet();
  const CastorCalibrations& getCalibrations(const DetId id) const;
  void setCalibrations(const DetId id, const CastorCalibrations& ca);
  void sort();
  void clear();
private:
  struct CalibSetObject {
    CalibSetObject(const DetId& aid) : id(aid) { }
    DetId id;
    CastorCalibrations calib;
    bool operator<(const CalibSetObject& cso) const { return id < cso.id; }
    bool operator==(const CalibSetObject& cso) const { return id == cso.id; }
  };
  typedef CalibSetObject Item;
  CastorCalibrations dummy;
  std::vector<CalibSetObject> mItems;
  bool sorted_;
};

#endif
