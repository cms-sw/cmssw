#ifndef CALIBFORMATS_HCALOBJECTS_HCALCALIBRATIONWIDTHSSET_H
#define CALIBFORMATS_HCALOBJECTS_HCALCALIBRATIONWIDTHSSET_H 1

#include "CalibFormats/CastorObjects/interface/CastorCalibrationWidths.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <vector>

/** \class CastorCalibrationWidthsSet
  *  
  * $Date: 2008/11/08 21:19:28 $
  * $Revision: 1.1 $
  * \author R.Ofierzynski
  * Adapted for CASTOR by L. Mundim
  */
class CastorCalibrationWidthsSet {
public:
  CastorCalibrationWidthsSet();
  const CastorCalibrationWidths& getCalibrationWidths(const DetId id) const;
  void setCalibrationWidths(const DetId id, const CastorCalibrationWidths& ca);
  void sort();
  void clear();
private:
  struct CalibWidthSetObject {
    CalibWidthSetObject(const DetId& aid) : id(aid) { }
    DetId id;
    CastorCalibrationWidths calib;
    bool operator<(const CalibWidthSetObject& cso) const { return id < cso.id; }
    bool operator==(const CalibWidthSetObject& cso) const { return id == cso.id; }
  };
  typedef CalibWidthSetObject Item;
  CastorCalibrationWidths dummy;
  std::vector<CalibWidthSetObject> mItems;
  bool sorted_;
};

#endif
