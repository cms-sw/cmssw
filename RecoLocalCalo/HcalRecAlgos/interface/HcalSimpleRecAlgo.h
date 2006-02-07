#ifndef HCALSIMPLERECALGO_H
#define HCALSIMPLERECALGO_H 1

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

/** \class HcalSimplerecAlgo
    
   $Date: 2005/11/28 19:56:27 $
   $Revision: 1.3 $
   \author J. Mans - Minnesota
*/
class HcalSimpleRecAlgo {
public:
  HcalSimpleRecAlgo(int firstSample, int samplesToAdd, bool correctForTimeslew);
  HBHERecHit reconstruct(const HBHEDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const;
  HFRecHit reconstruct(const HFDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const;
  HORecHit reconstruct(const HODataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const;
private:
  int firstSample_, samplesToAdd_;
  bool correctForTimeslew_;
};

#endif
