#ifndef RUNLIST_H
#define RUNLIST_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

typedef int run_t;

class RunList  : public IDBObject {
 public:
  friend class EcalCondDBInterface;

  RunList();
  ~RunList();

  // Methods for user data
  
  void setRunTag(RunTag tag);
  RunTag getRunTag() const;
  std::vector<RunIOV> getRuns() ;
  
  // Methods from IUniqueDBObject
  void fetchRuns() throw(std::runtime_error);


 private:
  // User data for this IOV
  std::vector<RunIOV> m_vec_runiov;
  RunTag m_runTag;

};

#endif
