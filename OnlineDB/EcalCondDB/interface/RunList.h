#ifndef RUNLIST_H
#define RUNLIST_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/LocationDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"

typedef int run_t;

class RunList : public IDBObject {
public:
  friend class EcalCondDBInterface;

  RunList();
  ~RunList() override;

  // Methods for user data

  void setRunTag(const RunTag& tag);
  RunTag getRunTag() const;
  std::vector<RunIOV> getRuns();

  // Methods from IUniqueDBObject
  void fetchRuns() noexcept(false);
  void fetchNonEmptyRuns() noexcept(false);
  void fetchNonEmptyGlobalRuns() noexcept(false);
  void fetchNonEmptyRuns(int min_run, int max_run) noexcept(false);
  void fetchNonEmptyGlobalRuns(int min_run, int max_run) noexcept(false);
  void fetchRuns(int min_run, int max_run) noexcept(false);
  void fetchRuns(int min_run, int max_run, bool withTriggers, bool withGlobalTriggers) noexcept(false);
  void fetchLastNRuns(int max_run, int n_runs) noexcept(false);
  void fetchRunsByLocation(int min_run, int max_run, const LocationDef& locDef) noexcept(false);
  void fetchGlobalRunsByLocation(int min_run, int max_run, const LocationDef& locDef) noexcept(false);

private:
  // User data for this IOV
  std::vector<RunIOV> m_vec_runiov;
  RunTag m_runTag;
};

#endif
