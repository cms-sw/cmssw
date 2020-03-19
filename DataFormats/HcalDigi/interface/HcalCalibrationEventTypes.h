#ifndef DATAFORMATS_HCALDIGI_HCALCALIBRATIONEVENTTYPES_H
#define DATAFORMATS_HCALDIGI_HCALCALIBRATIONEVENTTYPES_H 1

enum HcalCalibrationEventType {
  hc_Null = 0,
  hc_Pedestal = 1,
  hc_RADDAM = 2,
  hc_HBHEHPD = 3,
  hc_HOHPD = 4,
  hc_HFPMT = 5,
  hc_ZDC = 6
};

#endif
