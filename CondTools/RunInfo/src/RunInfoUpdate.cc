#include "CondTools/RunInfo/interface/RunInfoUpdate.h"
#include "CondCore/CondDB/interface/Session.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace {
  boost::posix_time::ptime parseTimeFromIsoString(const std::string& isoString) {
    boost::posix_time::time_input_facet* tif = new boost::posix_time::time_input_facet;
    tif->set_iso_extended_format();
    std::istringstream iss(isoString);
    iss.imbue(std::locale(std::locale::classic(), tif));
    boost::posix_time::ptime ret;
    iss >> ret;
    return ret;
  }

  void getRunTimeParams(const RunInfo& runInfo, boost::posix_time::ptime& start, boost::posix_time::ptime& end) {
    std::string startStr = runInfo.m_start_time_str;
    if (startStr != "null") {
      start = parseTimeFromIsoString(startStr);
    }
    end = start;
    std::string stopStr = runInfo.m_stop_time_str;
    if (stopStr != "null") {
      end = parseTimeFromIsoString(stopStr);
    }
  }
}  // namespace

RunInfoUpdate::RunInfoUpdate(cond::persistency::Session& dbSession) : m_dbSession(dbSession) {}

RunInfoUpdate::~RunInfoUpdate() {}

void RunInfoUpdate::appendNewRun(const RunInfo& runInfo) {
  cond::persistency::RunInfoEditor runInfoWriter = m_dbSession.editRunInfo();
  boost::posix_time::ptime start;
  boost::posix_time::ptime end;
  getRunTimeParams(runInfo, start, end);
  edm::LogInfo("RunInfoUpdate") << "[RunInfoUpdate::" << __func__ << "]: Checking run " << runInfo.m_run
                                << " for insertion in Condition DB" << std::endl;
  runInfoWriter.insertNew(runInfo.m_run, start, end);
  size_t newRuns = runInfoWriter.flush();
  edm::LogInfo("RunInfoUpdate") << "[RunInfoUpdate::" << __func__ << "]: " << newRuns << " new run(s) inserted."
                                << std::endl;
}

// only used in import command tool
size_t RunInfoUpdate::import(size_t maxEntries,
                             const std::string& sourceTag,
                             cond::persistency::Session& sourceSession) {
  cond::persistency::RunInfoEditor editor;
  std::cout << "# Loading tag " << sourceTag << "..." << std::endl;
  cond::persistency::IOVProxy runInfoTag = sourceSession.readIov(sourceTag, true);
  editor = m_dbSession.editRunInfo();
  cond::Time_t lastRun = editor.getLastInserted();
  std::cout << "# Last run found in RunInfo db : " << lastRun << std::endl;
  cond::persistency::IOVProxy::Iterator it = runInfoTag.begin();
  if (lastRun > 0) {
    it = runInfoTag.find(lastRun + 1);
  }
  if (it == runInfoTag.end() || (*it).since == lastRun) {
    std::cout << "# No more run found to be imported." << std::endl;
    return 0;
  }
  size_t n_entries = 0;
  while (it != runInfoTag.end() && n_entries <= maxEntries) {
    auto h = (*it).payloadId;
    std::shared_ptr<RunInfo> runInfo = sourceSession.fetchPayload<RunInfo>(h);
    if (runInfo->m_run != -1) {
      n_entries++;
      std::cout << "# Inserting run #" << runInfo->m_run << " (from since=" << (*it).since << ")" << std::endl;
      boost::posix_time::ptime start;
      boost::posix_time::ptime end;
      getRunTimeParams(*runInfo, start, end);
      editor.insert(runInfo->m_run, start, end);
    } else {
      std::cout << "# Skipping fake run #" << std::endl;
    }
    it++;
  }
  editor.flush();
  return n_entries;
}
