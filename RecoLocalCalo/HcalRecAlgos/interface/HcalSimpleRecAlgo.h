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
    
   $Date: $
   $Revision: $
   \author J. Mans - Minnesota
*/
class HcalSimpleRecAlgo {
public:
  HcalSimpleRecAlgo(int firstSample, int samplesToAdd);
  cms::HBHERecHit reconstruct(const cms::HBHEDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const;
  cms::HFRecHit reconstruct(const cms::HFDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const;
  cms::HORecHit reconstruct(const cms::HODataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const;
private:
  int firstSample_, samplesToAdd_;
};
#endif
