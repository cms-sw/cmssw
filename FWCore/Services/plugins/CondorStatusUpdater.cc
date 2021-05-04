
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Utilities/interface/TimingServiceBase.h"
#include "FWCore/Utilities/interface/CPUServiceBase.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/XrdAdaptor/src/XrdStatistics.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <spawn.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <sstream>
#include <atomic>
#include <string>
#include <set>

namespace edm {

  namespace service {

    class CondorStatusService {
    public:
      explicit CondorStatusService(ParameterSet const &pset, edm::ActivityRegistry &ar);
      ~CondorStatusService() {}
      CondorStatusService(const CondorStatusService &) = delete;
      CondorStatusService &operator=(const CondorStatusService &) = delete;

      static void fillDescriptions(ConfigurationDescriptions &descriptions);

    private:
      bool isChirpSupported();
      template <typename T>
      bool updateChirp(const std::string &key_suffix, const T &value);
      bool updateChirpQuoted(const std::string &key_suffix, const std::string &value);
      bool updateChirpImpl(std::string const &key, std::string const &value);
      inline void update();
      void firstUpdate();
      void lastUpdate();
      void updateImpl(time_t secsSinceLastUpdate);

      void preSourceConstruction(ModuleDescription const &md, int maxEvents, int maxLumis, int maxSecondsUntilRampdown);
      void eventPost(StreamContext const &iContext);
      void lumiPost(GlobalContext const &);
      void runPost(GlobalContext const &);
      void beginPre(PathsAndConsumesOfModulesBase const &, ProcessContext const &processContext);
      void beginPost();
      void endPost();
      void filePost(std::string const &, bool);

      bool m_debug;
      std::atomic_flag m_shouldUpdate;
      time_t m_beginJob = 0;
      time_t m_updateInterval = m_defaultUpdateInterval;
      float m_emaInterval = m_defaultEmaInterval;
      float m_rate = 0;
      static constexpr float m_defaultEmaInterval = 15 * 60;  // Time in seconds to average EMA over for event rate.
      static constexpr unsigned int m_defaultUpdateInterval = 3 * 60;
      std::atomic<time_t> m_lastUpdate;
      std::atomic<std::uint_least64_t> m_events;
      std::atomic<std::uint_least64_t> m_lumis;
      std::atomic<std::uint_least64_t> m_runs;
      std::atomic<std::uint_least64_t> m_files;
      std::string m_tag;
      edm::ParameterSetID m_processParameterSetID;

      std::uint_least64_t m_lastEventCount = 0;
    };

  }  // namespace service

}  // namespace edm

using namespace edm::service;

const unsigned int CondorStatusService::m_defaultUpdateInterval;
constexpr float CondorStatusService::m_defaultEmaInterval;

CondorStatusService::CondorStatusService(ParameterSet const &pset, edm::ActivityRegistry &ar)
    : m_debug(false), m_lastUpdate(0), m_events(0), m_lumis(0), m_runs(0), m_files(0) {
  m_shouldUpdate.clear();
  if (pset.exists("debug")) {
    m_debug = true;
  }
  if (!isChirpSupported()) {
    return;
  }

  firstUpdate();

  ar.watchPostCloseFile(this, &CondorStatusService::filePost);
  ar.watchPostEvent(this, &CondorStatusService::eventPost);
  ar.watchPostGlobalEndLumi(this, &CondorStatusService::lumiPost);
  ar.watchPostGlobalEndRun(this, &CondorStatusService::runPost);
  ar.watchPreBeginJob(this, &CondorStatusService::beginPre);
  ar.watchPostBeginJob(this, &CondorStatusService::beginPost);
  ar.watchPostEndJob(this, &CondorStatusService::endPost);

  if (pset.exists("updateIntervalSeconds")) {
    m_updateInterval = pset.getUntrackedParameter<unsigned int>("updateIntervalSeconds");
  }
  if (pset.exists("EMAInterval")) {
    m_emaInterval = pset.getUntrackedParameter<double>("EMAInterval");
  }
  if (pset.exists("tag")) {
    m_tag = pset.getUntrackedParameter<std::string>("tag");
  }
}

void CondorStatusService::eventPost(StreamContext const &iContext) {
  m_events++;
  update();
}

void CondorStatusService::lumiPost(GlobalContext const &) {
  m_lumis++;
  update();
}

void CondorStatusService::runPost(GlobalContext const &) {
  m_runs++;
  update();
}

void CondorStatusService::filePost(std::string const & /*lfn*/, bool /*usedFallback*/) {
  m_files++;
  update();
}

void CondorStatusService::beginPre(PathsAndConsumesOfModulesBase const &, ProcessContext const &processContext) {
  if (!m_processParameterSetID.isValid()) {
    m_processParameterSetID = processContext.parameterSetID();
  }
}

void CondorStatusService::beginPost() {
  ParameterSet const &processParameterSet = edm::getParameterSet(m_processParameterSetID);
  const edm::ParameterSet &pset = processParameterSet.getParameterSet("@main_input");
  // PSet info from edm::ScheduleItems
  int maxEvents =
      processParameterSet.getUntrackedParameterSet("maxEvents", ParameterSet()).getUntrackedParameter<int>("input", -1);
  int maxLumis = processParameterSet.getUntrackedParameterSet("maxLuminosityBlocks", ParameterSet())
                     .getUntrackedParameter<int>("input", -1);

  // lumisToProcess from EventSkipperByID (PoolSource and similar)
  std::vector<edm::LuminosityBlockRange> toProcess = pset.getUntrackedParameter<std::vector<LuminosityBlockRange>>(
      "lumisToProcess", std::vector<LuminosityBlockRange>());
  edm::sortAndRemoveOverlaps(toProcess);
  uint64_t lumiCount = 0;
  for (auto const &range : toProcess) {
    if (range.startRun() != range.endRun()) {
      break;
    }
    if (range.endLumi() >= edm::LuminosityBlockID::maxLuminosityBlockNumber()) {
      break;
    }
    lumiCount += (range.endLumi() - range.startLumi());
  }
  // Handle sources deriving from ProducerSourceBase
  unsigned int eventsPerLumi = pset.getUntrackedParameter<unsigned int>("numberEventsInLuminosityBlock", 0);
  if ((lumiCount == 0) && (maxEvents > 0) && (eventsPerLumi > 0)) {
    lumiCount = static_cast<unsigned int>(std::ceil(static_cast<float>(maxEvents) / static_cast<float>(eventsPerLumi)));
  }

  std::vector<std::string> fileNames =
      pset.getUntrackedParameter<std::vector<std::string>>("fileNames", std::vector<std::string>());
  std::stringstream ss_max_files;
  ss_max_files << fileNames.size();
  updateChirp("MaxFiles", ss_max_files.str());

  if (lumiCount > 0) {
    if (maxLumis < 0) {
      maxLumis = lumiCount;
    }
    if (maxLumis > static_cast<int>(lumiCount)) {
      maxLumis = lumiCount;
    }
  }
  if (maxEvents > 0) {
    std::stringstream ss_max_events;
    ss_max_events << maxEvents;
    updateChirp("MaxEvents", ss_max_events.str());
  }
  if (maxLumis > 0) {
    std::stringstream ss_max_lumis;
    ss_max_lumis << maxLumis;
    updateChirp("MaxLumis", ss_max_lumis.str());
  }

  m_beginJob = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  update();
}

void CondorStatusService::endPost() { lastUpdate(); }

bool CondorStatusService::isChirpSupported() {
  if (m_debug) {
    return true;
  }

  return std::getenv("_CONDOR_CHIRP_CONFIG") && updateChirp("Elapsed", "0");
}

void CondorStatusService::firstUpdate() {
  // Note we always update all our statistics to 0 / false / -1
  // This allows us to overwrite the activities of a previous cmsRun process
  // within this HTCondor job.
  updateImpl(0);
  updateChirp("MaxFiles", "-1");
  updateChirp("MaxEvents", "-1");
  updateChirp("MaxLumis", "-1");
  updateChirp("Done", "false");

  edm::Service<edm::CPUServiceBase> cpusvc;
  std::string models;
  double avgSpeed;
  if (cpusvc.isAvailable() && cpusvc->cpuInfo(models, avgSpeed)) {
    updateChirpQuoted("CPUModels", models);
    updateChirp("CPUSpeed", avgSpeed);
  }
}

void CondorStatusService::lastUpdate() {
  time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  updateImpl(now - m_lastUpdate);
  updateChirp("Done", "true");
  edm::Service<edm::CPUServiceBase> cpusvc;
  if (!cpusvc.isAvailable()) {
    std::cout << "At post, CPU service is NOT available.\n";
  }
}

void CondorStatusService::update() {
  time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  if ((now - m_lastUpdate.load(std::memory_order_relaxed)) > m_updateInterval) {
    if (!m_shouldUpdate.test_and_set(std::memory_order_acquire)) {
      // Caught exception is rethrown
      CMS_SA_ALLOW try {
        time_t sinceLastUpdate = now - m_lastUpdate;
        m_lastUpdate = now;
        updateImpl(sinceLastUpdate);
        m_shouldUpdate.clear(std::memory_order_release);
      } catch (...) {
        m_shouldUpdate.clear(std::memory_order_release);
        throw;
      }
    }
  }
}

void CondorStatusService::updateImpl(time_t sinceLastUpdate) {
  time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  time_t jobTime = now - m_beginJob;

  edm::Service<edm::TimingServiceBase> timingsvc;
  if (timingsvc.isAvailable()) {
    updateChirp("TotalCPU", timingsvc->getTotalCPU());
  }

  updateChirp("LastUpdate", now);

  if (!m_events || (m_events > m_lastEventCount)) {
    updateChirp("Events", m_events);
  }

  updateChirp("Lumis", m_lumis);

  updateChirp("Runs", m_runs);

  updateChirp("Files", m_files);

  float ema_coeff = 1 - std::exp(-static_cast<float>(sinceLastUpdate) /
                                 std::max(std::min(m_emaInterval, static_cast<float>(jobTime)), 1.0f));
  if (sinceLastUpdate > 0) {
    updateChirp("Elapsed", jobTime);
    m_rate = ema_coeff * static_cast<float>(m_events - m_lastEventCount) / static_cast<float>(sinceLastUpdate) +
             (1.0 - ema_coeff) * m_rate;
    m_lastEventCount = m_events;
    updateChirp("EventRate", m_rate);
  }

  // If Xrootd was used, pull the statistics from there.
  edm::Service<XrdAdaptor::XrdStatisticsService> xrdsvc;
  if (xrdsvc.isAvailable()) {
    for (auto const &iter : xrdsvc->condorUpdate()) {
      std::string site = iter.first;
      site.erase(std::remove_if(site.begin(), site.end(), [](char x) { return !isalnum(x) && (x != '_'); }),
                 site.end());
      auto &iostats = iter.second;
      updateChirp("IOSite_" + site + "_ReadBytes", iostats.bytesRead);
      updateChirp("IOSite_" + site + "_ReadTimeMS",
                  std::chrono::duration_cast<std::chrono::milliseconds>(iostats.transferTime).count());
    }
  }

  // Update storage account information
  auto const &stats = StorageAccount::summary();
  uint64_t readOps = 0;
  uint64_t readVOps = 0;
  uint64_t readSegs = 0;
  uint64_t readBytes = 0;
  uint64_t readTimeTotal = 0;
  uint64_t writeBytes = 0;
  uint64_t writeTimeTotal = 0;
  const auto token = StorageAccount::tokenForStorageClassName("tstoragefile");
  for (const auto &storage : stats) {
    // StorageAccount records statistics for both the TFile layer and the
    // StorageFactory layer.  However, the StorageFactory statistics tend to
    // be more accurate as various backends may alter the incoming read requests
    // (such as when lazy-download is used).
    if (storage.first == token.value()) {
      continue;
    }
    for (const auto &counter : storage.second) {
      if (counter.first == static_cast<int>(StorageAccount::Operation::read)) {
        readOps += counter.second.successes;
        readSegs++;
        readBytes += counter.second.amount;
        readTimeTotal += counter.second.timeTotal;
      } else if (counter.first == static_cast<int>(StorageAccount::Operation::readv)) {
        readVOps += counter.second.successes;
        readSegs += counter.second.vector_count;
        readBytes += counter.second.amount;
        readTimeTotal += counter.second.timeTotal;
      } else if ((counter.first == static_cast<int>(StorageAccount::Operation::write)) ||
                 (counter.first == static_cast<int>(StorageAccount::Operation::writev))) {
        writeBytes += counter.second.amount;
        writeTimeTotal += counter.second.timeTotal;
      }
    }
  }
  updateChirp("ReadOps", readOps);
  updateChirp("ReadVOps", readVOps);
  updateChirp("ReadSegments", readSegs);
  updateChirp("ReadBytes", readBytes);
  updateChirp("ReadTimeMsecs", readTimeTotal / (1000 * 1000));
  updateChirp("WriteBytes", writeBytes);
  updateChirp("WriteTimeMsecs", writeTimeTotal / (1000 * 1000));
}

template <typename T>
bool CondorStatusService::updateChirp(const std::string &key_suffix, const T &value) {
  std::stringstream ss;
  ss << value;
  return updateChirpImpl(key_suffix, ss.str());
}

bool CondorStatusService::updateChirpQuoted(const std::string &key_suffix, const std::string &value) {
  std::string value_copy = value;
  // Remove double-quotes or the \ character (as it has special escaping semantics in ClassAds).
  // Make sure we have ASCII characters.
  // Otherwise, remainder is allowed (including tabs, newlines, single-quotes).
  value_copy.erase(
      remove_if(
          value_copy.begin(), value_copy.end(), [](const char &c) { return !isascii(c) || (c == '"') || (c == '\\'); }),
      value_copy.end());
  return updateChirpImpl(key_suffix, "\"" + value_copy + "\"");
}

bool CondorStatusService::updateChirpImpl(const std::string &key_suffix, const std::string &value) {
  std::stringstream ss;
  ss << "ChirpCMSSW" << m_tag << key_suffix;
  std::string key = ss.str();
  if (m_debug) {
    std::cout << "condor_chirp set_job_attr_delayed " << key << " " << value << std::endl;
  }
  int pid = 0;
  posix_spawn_file_actions_t file_actions;
  int devnull_fd = open("/dev/null", O_RDWR);
  if (devnull_fd == -1) {
    return false;
  }
  posix_spawn_file_actions_init(&file_actions);
  posix_spawn_file_actions_adddup2(&file_actions, devnull_fd, 1);
  posix_spawn_file_actions_adddup2(&file_actions, devnull_fd, 2);
  const std::string chirp_name = "condor_chirp";
  const std::string set_job_attr = "set_job_attr_delayed";
  std::vector<const char *> argv;
  argv.push_back(chirp_name.c_str());
  argv.push_back(set_job_attr.c_str());
  argv.push_back(key.c_str());
  argv.push_back(value.c_str());
  argv.push_back(nullptr);
  int status = posix_spawnp(&pid, "condor_chirp", &file_actions, nullptr, const_cast<char *const *>(&argv[0]), environ);
  close(devnull_fd);
  posix_spawn_file_actions_destroy(&file_actions);
  if (status) {
    return false;
  }
  while ((waitpid(pid, &status, 0) == -1) && errno == -EINTR) {
  }
  return status == 0;
}

void CondorStatusService::fillDescriptions(ConfigurationDescriptions &descriptions) {
  ParameterSetDescription desc;
  desc.setComment("Service to update HTCondor with the current CMSSW status.");
  desc.addOptionalUntracked<unsigned int>("updateIntervalSeconds", m_defaultUpdateInterval)
      ->setComment("Interval, in seconds, for HTCondor updates");
  desc.addOptionalUntracked<bool>("debug", false)->setComment("Enable debugging of this service");
  desc.addOptionalUntracked<double>("EMAInterval", m_defaultEmaInterval)
      ->setComment("Interval, in seconds, to calculate event rate over (using EMA)");
  desc.addOptionalUntracked<std::string>("tag")->setComment(
      "Identifier tag for this process (a value of 'Foo' results in ClassAd attributes of the form 'ChirpCMSSWFoo*')");
  descriptions.add("CondorStatusService", desc);
}

typedef edm::serviceregistry::AllArgsMaker<edm::service::CondorStatusService> CondorStatusServiceMaker;
DEFINE_FWK_SERVICE_MAKER(CondorStatusService, CondorStatusServiceMaker);
