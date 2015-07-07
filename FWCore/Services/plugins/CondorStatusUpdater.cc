
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <spawn.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <sstream>
#include <atomic>
#include <string>

namespace edm {

    namespace service {

        class CondorStatusService
        {

        public:

            explicit CondorStatusService(ParameterSet const& pset, edm::ActivityRegistry& ar);
            ~CondorStatusService() {}
            CondorStatusService(const CondorStatusService&) = delete;
            CondorStatusService& operator=(const CondorStatusService&) = delete;

            static void fillDescriptions(ConfigurationDescriptions &descriptions);

        private:

            bool isChirpSupported();
            bool updateChirp(std::string const &key, std::string const &value);
            inline void update();
            void firstUpdate();
            void lastUpdate();
            void updateImpl(time_t secsSinceLastUpdate);

            void preSourceConstruction(ModuleDescription const &md, int maxEvents, int maxLumis, int maxSecondsUntilRampdown);
            void eventPost(StreamContext const& iContext);
            void lumiPost(GlobalContext const&);
            void runPost(GlobalContext const&);
            void beginPre(PathsAndConsumesOfModulesBase const&, ProcessContext const& processContext);
            void beginPost();
            void endPost();
            void filePost(std::string const &, bool);

            bool m_debug;
            std::atomic_flag m_shouldUpdate;
            time_t m_beginJob = 0;
            time_t m_updateInterval = m_defaultUpdateInterval;
            float m_emaInterval = m_defaultEmaInterval;
            float m_rate = 0;
            static constexpr float m_defaultEmaInterval = 15*60; // Time in seconds to average EMA over for event rate.
            static constexpr unsigned int m_defaultUpdateInterval = 3*60;
            std::atomic<time_t> m_lastUpdate;
            std::atomic<std::uint_least64_t> m_events;
            std::atomic<std::uint_least64_t> m_lumis;
            std::atomic<std::uint_least64_t> m_runs;
            std::atomic<std::uint_least64_t> m_files;
            edm::ParameterSetID m_processParameterSetID;

            std::uint_least64_t m_lastEventCount = 0;
        };

    }

}

using namespace edm::service;

CondorStatusService::CondorStatusService(ParameterSet const& pset, edm::ActivityRegistry& ar)
  :
    m_debug(false),
    m_lastUpdate(0),
    m_events(0),
    m_lumis(0),
    m_runs(0),
    m_files(0)
{
    m_shouldUpdate.clear();
    if (pset.exists("debug"))
    {
        m_debug = true;
    }
    if (!isChirpSupported()) {return;}

    firstUpdate();

    ar.watchPostCloseFile(this, &CondorStatusService::filePost);
    ar.watchPostEvent(this, &CondorStatusService::eventPost);
    ar.watchPostGlobalEndLumi(this, &CondorStatusService::lumiPost);
    ar.watchPostGlobalEndRun(this, &CondorStatusService::runPost);
    ar.watchPreBeginJob(this, &CondorStatusService::beginPre);
    ar.watchPostBeginJob(this, &CondorStatusService::beginPost);
    ar.watchPostEndJob(this, &CondorStatusService::endPost);

    if (pset.exists("updateIntervalSeconds"))
    {
        m_updateInterval = pset.getUntrackedParameter<unsigned int>("updateIntervalSeconds");
    }
    if (pset.exists("EMAInterval"))
    {
        m_emaInterval = pset.getUntrackedParameter<double>("EMAInterval");
    }
}


void
CondorStatusService::eventPost(StreamContext const& iContext)
{
    m_events++;
    update();
}


void
CondorStatusService::lumiPost(GlobalContext const&)
{
    m_lumis++;
    update();
}


void
CondorStatusService::runPost(GlobalContext const&)
{
    m_runs++;
    update();
}


void
CondorStatusService::filePost(std::string const & /*lfn*/, bool /*usedFallback*/)
{
    m_files++;
    update();
}


void
CondorStatusService::beginPre(PathsAndConsumesOfModulesBase const&, ProcessContext const& processContext)
{
    if (!m_processParameterSetID.isValid())
    {
        m_processParameterSetID = processContext.parameterSetID();
    }
}


void
CondorStatusService::beginPost()
{
    ParameterSet const& processParameterSet = edm::getParameterSet(m_processParameterSetID);
    const edm::ParameterSet &pset = processParameterSet.getParameterSet("@main_input");
    // PSet info from edm::ScheduleItems
    int maxEvents = processParameterSet.getUntrackedParameterSet("maxEvents", ParameterSet()).getUntrackedParameter<int>("input", -1);
    int maxLumis = processParameterSet.getUntrackedParameterSet("maxLuminosityBlocks", ParameterSet()).getUntrackedParameter<int>("input", -1);

    // lumisToProcess from EventSkipperByID (PoolSource and similar)
    std::vector<edm::LuminosityBlockRange> toProcess = pset.getUntrackedParameter<std::vector<LuminosityBlockRange> >("lumisToProcess", std::vector<LuminosityBlockRange>());
    edm::sortAndRemoveOverlaps(toProcess);
    uint64_t lumiCount = 0;
    for (auto const &range : toProcess)
    {
        if (range.startRun() != range.endRun()) {break;}
        if (range.endLumi() >= edm::LuminosityBlockID::maxLuminosityBlockNumber()) {break;}
        lumiCount += (range.endLumi()-range.startLumi());
    }
    // Handle sources deriving from ProducerSourceBase
    unsigned int eventsPerLumi = pset.getUntrackedParameter<unsigned int>("numberEventsInLuminosityBlock", 0);
    if ((lumiCount == 0) && (maxEvents > 0) && (eventsPerLumi > 0))
    {
        lumiCount = static_cast<unsigned int>(std::ceil(static_cast<float>(maxEvents) / static_cast<float>(eventsPerLumi)));
    }

    std::vector<std::string> fileNames = pset.getUntrackedParameter<std::vector<std::string>>("fileNames", std::vector<std::string>());
    std::stringstream ss_max_files; ss_max_files << fileNames.size();
    updateChirp("ChirpCMSSWMaxFiles", ss_max_files.str());

    if (lumiCount > 0)
    {
        if (maxLumis < 0) {maxLumis = lumiCount;}
        if (maxLumis > static_cast<int>(lumiCount))
        {
            maxLumis = lumiCount;
        }
    }
    if (maxEvents > 0)
    {
        std::stringstream ss_max_events; ss_max_events << maxEvents;
        updateChirp("ChirpCMSSWMaxEvents", ss_max_events.str());
    }
    if (maxLumis > 0)
    {
        std::stringstream ss_max_lumis; ss_max_lumis << maxLumis;
        updateChirp("ChirpCMSSWMaxLumis", ss_max_lumis.str());
    }

    m_beginJob = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    update();
}


void
CondorStatusService::endPost()
{
    lastUpdate();
}


bool
CondorStatusService::isChirpSupported()
{
    if (m_debug) {return true;}

    return getenv("_CONDOR_CHIRP_CONFIG") && updateChirp("ChirpCMSSWElapsed", "0");
}


void
CondorStatusService::firstUpdate()
{
    // Note we always update all our statistics to 0 / false / -1
    // This allows us to overwrite the activities of a previous cmsRun process
    // within this HTCondor job.
    updateImpl(0);
    updateChirp("ChirpCMSSWMaxFiles", "-1");
    updateChirp("ChirpCMSSWMaxEvents", "-1");
    updateChirp("ChirpCMSSWMaxLumis", "-1");
    updateChirp("ChirpCMSSWDone", "false");
}


void
CondorStatusService::lastUpdate()
{
    time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    updateImpl(now - m_lastUpdate);
    updateChirp("ChirpCMSSWDone", "true");
}


void
CondorStatusService::update()
{
    time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    if ((now - m_lastUpdate.load(std::memory_order_relaxed)) > m_updateInterval)
    {
        if (!m_shouldUpdate.test_and_set(std::memory_order_acquire))
        {
            try
            {
                time_t sinceLastUpdate = now - m_lastUpdate;
                m_lastUpdate = now;
                updateImpl(sinceLastUpdate);
                m_shouldUpdate.clear(std::memory_order_release);
            }
            catch (...)
            {
                m_shouldUpdate.clear(std::memory_order_release);
                throw;
            }
        }
    }
}


void
CondorStatusService::updateImpl(time_t sinceLastUpdate)
{
    time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    time_t jobTime = now-m_beginJob;

    std::stringstream ss_now; ss_now << now;
    updateChirp("ChirpCMSSWLastUpdate", ss_now.str());

    if (m_events > m_lastEventCount)
    {
        std::stringstream ss_events; ss_events << m_events;
        updateChirp("ChirpCMSSWEvents", ss_events.str());
    }

    std::stringstream ss_lumis; ss_lumis << m_lumis;
    updateChirp("ChirpCMSSWLumis", ss_lumis.str());

    std::stringstream ss_runs; ss_runs << m_runs;
    updateChirp("ChirpCMSSWRuns", ss_runs.str());

    std::stringstream ss_files; ss_files << m_files;
    updateChirp("ChirpCMSSWFiles", ss_files.str());

    float ema_coeff = 1 - std::exp(-static_cast<float>(sinceLastUpdate)/m_emaInterval);
    if (sinceLastUpdate > 0)
    {
        std::stringstream ss_elapsed; ss_elapsed << jobTime;
        updateChirp("ChirpCMSSWElapsed", ss_elapsed.str());
        m_rate = ema_coeff*static_cast<float>(m_events-m_lastEventCount)/static_cast<float>(sinceLastUpdate) + (1.0-ema_coeff)*m_rate;
        m_lastEventCount = m_events;
        std::stringstream ss_rate; ss_rate << m_rate;
        updateChirp("ChirpCMSSWEventRate", ss_rate.str());
    }
}


bool
CondorStatusService::updateChirp(const std::string &key, const std::string &value)
{
    if (m_debug)
    {
        std::cout << "condor_chirp set_job_attr_delayed " << key << " " << value << std::endl;
    }
    int pid = 0;
    posix_spawn_file_actions_t file_actions;
    int devnull_fd = open("/dev/null", O_RDWR);
    if (devnull_fd == -1) {return false;}
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
    argv.push_back(NULL);
    int status = posix_spawnp(&pid, "condor_chirp", &file_actions, NULL, const_cast<char* const*>(&argv[0]), environ);
    close(devnull_fd);
    posix_spawn_file_actions_destroy(&file_actions);
    if (status)
    {
       return false;
    }
    while ((waitpid(pid, &status, 0) == -1) && errno == -EINTR) {}
    return status == 0;
}


void
CondorStatusService::fillDescriptions(ConfigurationDescriptions &descriptions)
{
    ParameterSetDescription desc;
    desc.setComment("Service to update HTCondor with the current CMSSW status.");
    desc.addOptionalUntracked<unsigned int>("updateIntervalSeconds", m_defaultUpdateInterval)
      ->setComment("Interval, in seconds, for HTCondor updates");
    desc.addOptionalUntracked<bool>("debug", false)
      ->setComment("Enable debugging of this service");
    desc.addOptionalUntracked<double>("EMAInterval", m_defaultEmaInterval)
      ->setComment("Interval, in seconds, to calculate event rate over (using EMA)");
    descriptions.add("CondorStatusService", desc);
}


typedef edm::serviceregistry::AllArgsMaker<edm::service::CondorStatusService> CondorStatusServiceMaker;
DEFINE_FWK_SERVICE_MAKER(CondorStatusService,CondorStatusServiceMaker);

