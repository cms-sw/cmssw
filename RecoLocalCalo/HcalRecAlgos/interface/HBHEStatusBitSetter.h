#ifndef HBHESTATUSBITSETTER_H
#define HBHESTATUSBITSETTER_H 1


#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "CondFormats/HcalObjects/interface/HcalFrontEndMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

class HBHEStatusBitSetter {
public:
  HBHEStatusBitSetter();
  HBHEStatusBitSetter(double nominalPedestal,double hitEnergyMinimum,int hitMultiplicityThreshold,const std::vector<edm::ParameterSet>& pulseShapeParameterSets);
  ~HBHEStatusBitSetter();

  void SetFrontEndMap(const HcalFrontEndMap* m); 
  void Clear();
  void SetFlagsFromDigi(HBHERecHit& hbhe, const HBHEDataFrame& digi,
                        const HcalCoder& coder, const HcalCalibrations& calib);
  void rememberHit(const HBHERecHit& hbhe);
  void SetFlagsFromRecHits(HBHERecHitCollection& rec);

private:
  HBHEStatusBitSetter(const HBHEStatusBitSetter&) = delete;
  HBHEStatusBitSetter& operator=(const HBHEStatusBitSetter&) = delete;
  
  double hitEnergyMinimum_;
  int hitMultiplicityThreshold_;
  double nominalPedestal_;
  const HcalFrontEndMap *frontEndMap_;
  std::vector<int> hpdMultiplicity_;
  std::vector< std::vector<double> > pulseShapeParameters_;
};

#endif
