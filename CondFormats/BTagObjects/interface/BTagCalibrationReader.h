#ifndef BTagCalibrationReader_H
#define BTagCalibrationReader_H

/**
 * BTagCalibrationReader
 *
 * Helper class to pull out a specific set of BTagEntry's out of a
 * BTagCalibration. TF1 functions are set up at initialization time.
 *
 ************************************************************/

#include <map>
#include <string>
#include <vector>
#include <TF1.h>

#include "CondFormats/BTagObjects/interface/BTagEntry.h"
#include "CondFormats/BTagObjects/interface/BTagCalibration.h"

class BTagCalibrationReader
{
public:
  BTagCalibrationReader() {}
  BTagCalibrationReader(BTagCalibration& c, BTagEntry::Parameters p);
  BTagCalibrationReader(BTagCalibration& c,
                        BTagEntry::OperatingPoint op,
                        BTagEntry::JetFlavor jf,
                        std::string measurementType="comb",
                        std::string sysType="central");
  ~BTagCalibrationReader() {}

  double eval(float eta, float pt, float discr=0.) const;

protected:
  struct TmpEntry {
    float etaMin;
    float etaMax;
    float ptMin;
    float ptMax;
    float discrMin;
    float discrMax;
    TF1 func;
  };
  void setupTmpData(BTagCalibration& c);

  BTagEntry::Parameters params;
  std::vector<TmpEntry> tmpData_;

  COND_SERIALIZABLE;
};

#endif  // BTagCalibrationReader_H