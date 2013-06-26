#ifndef HCALTIMINGCORRECTOR_H
#define HCALTIMINGCORRECTOR_H


#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"

class HcalTimingCorrector {
 public:
  HcalTimingCorrector();
  ~HcalTimingCorrector();
  static void Correct(HBHERecHit& rechit, const HBHEDataFrame& digi, int favorite_capid);
  static void Correct(HORecHit&   rechit, const HODataFrame&   digi, int favorite_capid);
  static void Correct(HFRecHit&   rechit, const HFDataFrame&   digi, int favorite_capid);

};

#endif
