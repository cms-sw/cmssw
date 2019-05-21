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

class MonRunList : public IDBObject {
public:
  friend class EcalCondDBInterface;

  MonRunList();
  ~MonRunList() override;

  // Methods for user data

  void setRunTag(const RunTag& tag);
  RunTag getRunTag() const;
  void setMonRunTag(const MonRunTag& tag);
  MonRunTag getMonRunTag() const;
  std::vector<MonRunIOV> getRuns();

  // Methods from IUniqueDBObject
  void fetchRuns() noexcept(false);
  void fetchRuns(int min_run, int max_run) noexcept(false);
  void fetchLastNRuns(int max_run, int n_runs) noexcept(false);

private:
  // User data for this IOV
  std::vector<MonRunIOV> m_vec_monruniov;
  RunTag m_runTag;
  MonRunTag m_monrunTag;
};

#endif
