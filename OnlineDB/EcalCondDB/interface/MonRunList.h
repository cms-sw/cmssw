#ifndef MONRUNLIST_H
#define MONRUNLIST_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

typedef int run_t;

class MonRunList  : public IDBObject {
 public:
  friend class EcalCondDBInterface;

  MonRunList();
  ~MonRunList();

  // Methods for user data
  
  void setRunTag(RunTag tag);
  RunTag getRunTag() const;
  void setMonRunTag(MonRunTag tag);
  MonRunTag getMonRunTag() const;
  std::vector<MonRunIOV> getRuns() ;
  
  // Methods from IUniqueDBObject
  void fetchRuns() throw(std::runtime_error);
  void fetchRuns(int min_run, int max_run) throw(std::runtime_error);
  void fetchLastNRuns( int max_run, int n_runs  )throw(std::runtime_error);

 private:
  // User data for this IOV
  std::vector<MonRunIOV> m_vec_monruniov;
  RunTag m_runTag;
  MonRunTag m_monrunTag;

};

#endif
