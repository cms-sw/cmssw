#ifndef CondTools_RunInfo_RunInfoUpdate_h
#define CondTools_RunInfo_RunInfoUpdate_h

#include "CondFormats/RunInfo/interface/RunInfo.h"

namespace cond {
  namespace persistency {
    class Session;
  }
}  // namespace cond

class RunInfoUpdate {
public:
  explicit RunInfoUpdate(cond::persistency::Session& dbSession);

  ~RunInfoUpdate();
  void appendNewRun(const RunInfo& run);

  size_t import(size_t maxEntries, const std::string& tag, cond::persistency::Session& sourceSession);

private:
  cond::persistency::Session& m_dbSession;
};

#endif
