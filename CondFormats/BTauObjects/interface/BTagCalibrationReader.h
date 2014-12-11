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

#include "CondFormats/BTauObjects/interface/BTagEntry.h"
#include "CondFormats/BTauObjects/interface/BTagCalibration.h"

class BTagCalibrationReader
{
public:
  BTagCalibrationReader() {}
  BTagCalibrationReader(const BTagCalibration* c,
                        BTagEntry::OperatingPoint op,
                        std::string measurementType="comb",
                        std::string sysType="central");
  ~BTagCalibrationReader() {}

  double eval(BTagEntry::JetFlavor jf,
              float eta,
              float pt,
              float discr=0.) const;

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
  void setupTmpData(const BTagCalibration* c);

  BTagEntry::Parameters params;
  std::map<BTagEntry::JetFlavor, std::vector<TmpEntry> > tmpData_;
  std::vector<bool> useAbsEta;
};

#endif  // BTagCalibrationReader_H
