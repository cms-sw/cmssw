#ifndef LMFRUNLIST_H
#define LMFRUNLIST_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"

typedef int run_t;

class LMFRunList  : public IDBObject {
 public:
  friend class EcalCondDBInterface;

  LMFRunList();
  ~LMFRunList();

  // Methods for user data
  
  void setRunTag(RunTag tag);
  RunTag getRunTag() const;
  void setLMFRunTag(LMFRunTag tag);
  LMFRunTag getLMFRunTag() const;
  std::vector<LMFRunIOV> getRuns() ;
  
  // Methods from IUniqueDBObject
  void fetchRuns() throw(std::runtime_error);
  void fetchRuns(int min_run, int max_run) throw(std::runtime_error);
  void fetchLastNRuns( int max_run, int n_runs  )throw(std::runtime_error);
  void fetchRuns(uint64_t start_micro, int end_run) throw(runtime_error);
  void fetchRuns(uint64_t start_micro, int min_run, int end_run) throw(runtime_error);
  void fetchLastNRunsBefore(uint64_t start_micro , int n_runs  ) throw(runtime_error);


 private:
  // User data for this IOV
  std::vector<LMFRunIOV> m_vec_lmfruniov;
  RunTag m_runTag;
  LMFRunTag m_lmfrunTag;

};

#endif
