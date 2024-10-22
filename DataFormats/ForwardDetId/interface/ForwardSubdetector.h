#ifndef DataFormats_ForwardDetId_ForwardSubDetector_H
#define DataFormats_ForwardDetId_ForwardSubDetector_H 1

enum ForwardSubdetector {
  ForwardEmpty = 0,
  FastTime = 1,
  BHM = 2,
  HGCEE = 3,
  HGCHEF = 4,
  HGCHEB = 5,
  HFNose = 6,
  HGCTrigger = 7
};
enum HGCalTriggerSubdetector { HFNoseTrigger = 0, HGCalEETrigger = 1, HGCalHSiTrigger = 2, HGCalHScTrigger = 3 };

#endif
