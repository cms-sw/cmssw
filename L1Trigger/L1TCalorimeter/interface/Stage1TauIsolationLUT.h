// Stage1TauIsolationLUT.h
// Author: Leonard Apanasevich
//

#ifndef STAGE1TAUISOLATIONLUT_H
#define STAGE1TAUISOLATIONLUT_H

#define NBITS_JET_ET_LUT 8
#define NBITS_TAU_ET_LUT 8
#define NBITS_DATA 1   // number of bits in the payload
#define LUT_VERSION 1  // bump up the version number is any of the above is changed or if the relative tau iso algo is changed

#include <iostream>
#include <math.h>
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  class Stage1TauIsolationLUT{
  public:
    static const unsigned int nbitsJet; // number of bits used to store JET ET in LUT
    static const unsigned int nbitsTau; // number of bits used to store TAU ET in LUT
    static const unsigned int nbits_data; // number of bits in the payload
    static const unsigned int lut_version;

    Stage1TauIsolationLUT(CaloParamsHelper* params);
    virtual ~Stage1TauIsolationLUT();

    unsigned lutAddress(unsigned int, unsigned int) const;
    int lutPayload(unsigned int) const;
  private:

    CaloParamsHelper* const params_;
    //double tauMaxJetIsolationA;
    //double tauMaxJetIsolationB;
    //int tauMinPtJetIsolationB;

  };
  
}
#endif
