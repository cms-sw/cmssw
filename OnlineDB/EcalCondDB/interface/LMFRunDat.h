#ifndef LMFRUNDAT_H
#define LMFRUNDAT_H

#include <math.h>

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
*/

#include "OnlineDB/EcalCondDB/interface/LMFDat.h"

/**
 *   LMF_RUN_DAT interface
 */
class LMFRunDat : public LMFDat {
 public:
  LMFRunDat();
  LMFRunDat(EcalDBConnection *conn);
  LMFRunDat(oracle::occi::Environment* env,
	    oracle::occi::Connection* conn);
  ~LMFRunDat() { }

  int getEvents(const EcalLogicID &id) {
    return (int)rint(getData(id, "NEVENTS"));
  }
  int getQualityFlag(const EcalLogicID &id) {
    return (int)rint(getData(id, "QUALITY_FLAG"));
  }
  LMFRunDat& setEvents(const EcalLogicID &id, int n) {
    LMFDat::setData(id, "NEVENTS", (float)n);
    return *this;
  }
  LMFRunDat& setQualityFlag(const EcalLogicID &id, int q) {
    LMFDat::setData(id, "QUALITY_FLAG", (float)q);
    return *this;
  }
  LMFRunDat& setData(const EcalLogicID &id, int n, int q) {
    LMFDat::setData(id, "NEVENTS", (float)n);
    LMFDat::setData(id, "QUALITY_FLAG", (float)q);
    return *this;
  }
  LMFRunDat& Data(const EcalLogicID &id, const std::vector<float> &v) {
    LMFDat::setData(id, v);
    return *this;
  }

 protected:

};

#endif
