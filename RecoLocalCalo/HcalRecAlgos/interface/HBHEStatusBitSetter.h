#ifndef HBHESTATUSBITSETTER_H
#define HBHESTATUSBITSETTER_H 1


#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "CondTools/Hcal/interface/HcalLogicalMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HBHEStatusBitSetter {
 public:
  HBHEStatusBitSetter();
  HBHEStatusBitSetter(double nominalPedestal,double hitEnergyMinimum,int hitMultiplicityThreshold,std::vector<edm::ParameterSet> pulseShapeParameterSets);
  ~HBHEStatusBitSetter();
  void Clear();
  void SetFlagsFromDigi(HBHERecHit& hbhe, const HBHEDataFrame& digi);
  void SetFlagsFromRecHits(HBHERecHitCollection& rec);
 private:
  double hitEnergyMinimum_;
  int hitMultiplicityThreshold_;
  double nominalPedestal_;
  HcalLogicalMap *logicalMap_;
  std::vector<int> hpdMultiplicity_;
  std::vector< std::vector<double> > pulseShapeParameters_;
};

#endif
