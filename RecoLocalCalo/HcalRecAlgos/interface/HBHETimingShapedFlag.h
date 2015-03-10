#ifndef HBHETIMINGSHAPEDFLAG_GUARD_H
#define HBHETIMINGSHAPEDFLAG_GUARD_H

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"


class HBHETimingShapedFlagSetter {
 public:
  HBHETimingShapedFlagSetter();
  HBHETimingShapedFlagSetter(const std::vector<double>& tfilterEnvelope);
  HBHETimingShapedFlagSetter(const std::vector<double>& tfilterEnvelope,
			     bool ignorelowest,
			     bool ignorehighest,
			     double win_offset,
			     double win_gain);
  ~HBHETimingShapedFlagSetter();
  void Clear();

  void dumpInfo();

  // returns status suitable for flag setting
  // This routine made available for reflagger code
  //
  int timingStatus(const HBHERecHit& hbhe);

  // Sets "HBHETimingShapedCutsBits" field in response to output
  // from "timingStatus()"
  //
  void SetTimingShapedFlags(HBHERecHit& hbhe);

 private:
  // key   = integer GeV (to avoid FP issues),
  // value = low/high values for timing in ns
  //
  typedef std::map<int,std::pair<double,double> > TfilterEnvelope_t;
  TfilterEnvelope_t tfilterEnvelope_;

  void makeTfilterEnvelope(const std::vector<double>& v_userEnvelope);

  bool ignorelowest_;
  bool ignorehighest_;
  double win_offset_;
  double win_gain_;
};

#endif
