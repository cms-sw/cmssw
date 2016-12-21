#ifndef BTagCalibrationReader_H
#define BTagCalibrationReader_H

/**
 * BTagCalibrationReader
 *
 * Helper class to pull out a specific set of BTagEntry's out of a
 * BTagCalibration. TF1 functions are set up at initialization time.
 *
 ************************************************************/

#include <memory>
#include <string>

#include "CondFormats/BTauObjects/interface/BTagEntry.h"
#include "CondFormats/BTauObjects/interface/BTagCalibration.h"


class BTagCalibrationReader
{
public:
  BTagCalibrationReader() {}
  BTagCalibrationReader(BTagEntry::OperatingPoint op,
                        std::string sysType="central");

  void load(const BTagCalibration & c,
            BTagEntry::JetFlavor jf,
            std::string measurementType="comb");

  double eval(BTagEntry::JetFlavor jf,
              float eta,
              float pt,
              float discr=0.) const;

  std::pair<float, float> min_max_pt(BTagEntry::JetFlavor jf, 
                                     float eta, 
                                     float discr=0.) const;

protected:
  class BTagCalibrationReaderImpl;
  std::auto_ptr<BTagCalibrationReaderImpl> pimpl;
};


#endif  // BTagCalibrationReader_H
