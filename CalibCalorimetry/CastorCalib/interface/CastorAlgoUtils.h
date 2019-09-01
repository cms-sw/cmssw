#ifndef DQM_CASTORALGOUTILS_H
#define DQM_CASTORALGOUTILS_H

#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"

namespace reco {
  namespace castor {

    void getLinearizedADC(
        const CastorQIEShape& shape, const CastorQIECoder* coder, int bins, int capid, float& lo, float& hi);

    float maxDiff(float one, float two, float three, float four);

  }  // namespace castor
}  // namespace reco

#endif
