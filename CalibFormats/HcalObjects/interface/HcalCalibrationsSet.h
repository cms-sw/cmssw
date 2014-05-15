#ifndef CALIBFORMATS_HCALOBJECTS_HCALCALIBRATIONSSET_H
#define CALIBFORMATS_HCALOBJECTS_HCALCALIBRATIONSSET_H 1

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include <vector>

/** \class HcalCalibrationsSet
  *  
  * $Date: 2007/12/20 01:39:52 $
  * $Revision: 1.1 $
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
