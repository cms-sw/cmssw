#ifndef BTagEntry_H
#define BTagEntry_H

/**
 *
 * BTagEntry
 *
 * Represents one pt-dependent calibration function.
 *
 * measurement_type:    e.g. comb, ttbar, di-mu, boosted, ...
 * sys_type:            e.g. central, plus, minus, plus_JEC, plus_JER, ...
 *
 * Everything is converted into a function, as it is easiest to store it in a
 * txt or json file.
 *
 ************************************************************/

#include <string>
#include <TF1.h>
#include <TH1.h>

#include "CondFormats/Serialization/interface/Serializable.h"

class BTagEntry
{
public:
  enum OperatingPoint {
    OP_TIGHT=0,
    OP_MEDIUM=1,
    OP_LOOSE=2,
    OP_RESHAPING=3,
  };
  enum JetFlavor {
    FLAV_B=0,
    FLAV_C=1,
    FLAV_UDSG=2,
  };
  struct Parameters {
    // these go into the identifier token
    OperatingPoint operatingPoint;
    JetFlavor jetFlavor;
    std::string measurementType;
    std::string sysType;

    // these do _not_ go into the identifier token
    float etaMin;
    float etaMax;
    int reshapingBin;

    // default constructor
    Parameters(
      OperatingPoint op=OP_TIGHT, JetFlavor jf=FLAV_B,
      std::string measurement_type="comb", std::string sys_type="central",
      float etaMin=-99999., float etaMax=99999., int reshaping_bin=-1
    );

    // identifier token function
    std::string token();

    COND_SERIALIZABLE;
  };

  BTagEntry() {}
  BTagEntry(const std::string &func,
            Parameters p,
            float pt_min=0.,
            float pt_max=99999.);
  BTagEntry(const TF1* func, Parameters p);
  BTagEntry(const TH1* histo, Parameters p);
  ~BTagEntry() {}

  // public, no getters needed
  float ptMin, ptMax;
  std::string formula;
  Parameters params;

  COND_SERIALIZABLE;
};

#endif  // BTagEntry_H
