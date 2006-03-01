#ifndef LMFRUNIOV_H
#define LMFRUNIOV_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

typedef int subrun_t;

class LMFRunIOV : public IIOV {
 public:
  friend class EcalCondDBInterface;

  LMFRunIOV();
  ~LMFRunIOV();

  // Methods for user data
  void setLMFRunTag(LMFRunTag tag);
  LMFRunTag getLMFRunTag() const;
  void setRunIOV(RunIOV iov);
  RunIOV getRunIOV();
  void setSubRunNumber(subrun_t subrun);
  run_t getSubRunNumber() const;
  void setSubRunStart(Tm start);
  Tm getSubRunStart() const;
  void setSubRunEnd(Tm end);
  Tm getSubRunEnd() const;

  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

 private:
  // User data for this IOV
  LMFRunTag m_lmfRunTag;
  RunIOV m_runIOV;
  subrun_t m_subRunNum;
  Tm m_subRunStart;
  Tm m_subRunEnd;

  int writeDB() throw(std::runtime_error);
  void fetchParentIDs(int* lmfRunTagID, int* runIOVID) throw(std::runtime_error);

  void setByRun(LMFRunTag* lmftag, RunIOV* runiov, subrun_t subrun) throw(std::runtime_error);
};

#endif
