#ifndef DATAFORMATS_HCALDETID_HCALSUBDETECTOR_H
#define DATAFORMATS_HCALDETID_HCALSUBDETECTOR_H 1

enum HcalSubdetector {
  HcalEmpty = 0,
  HcalBarrel = 1,
  HcalEndcap = 2,
  HcalOuter = 3,
  HcalForward = 4,
  HcalTriggerTower = 5,
  HcalOther = 7
};

enum HcalOtherSubdetector {
  HcalOtherEmpty = 0,
  HcalCalibration = 2,
  HcalDcsBarrel = 3,
  HcalDcsEndcap = 4,
  HcalDcsOuter = 5,
  HcalDcsForward = 6
};

#endif
