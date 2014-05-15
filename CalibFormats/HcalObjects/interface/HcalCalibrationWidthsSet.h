#ifndef CALIBFORMATS_HCALOBJECTS_HCALCALIBRATIONWIDTHSSET_H
#define CALIBFORMATS_HCALOBJECTS_HCALCALIBRATIONWIDTHSSET_H 1

#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include <vector>

/** \class HcalCalibrationWidthsSet
  *  
  * $Date: 2008/11/08 21:19:28 $
  * $Revision: 1.1 $
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
      if (aid.det()==DetId::Hcal) {
	HcalDetId hcid(aid);
	id   = HcalDetId(hcid.subdet(),hcid.ieta(),hcid.iphi(),hcid.depth());
      } else if (aid.det()==DetId::Calo && aid.subdetId()==HcalZDCDetId::SubdetectorId) {
	HcalZDCDetId hcid(aid);
	id   = HcalZDCDetId(hcid.section(),(hcid.zside()>0),hcid.channel());
      } else {
	id   = aid;
      }
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
