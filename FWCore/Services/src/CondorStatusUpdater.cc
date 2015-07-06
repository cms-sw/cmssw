
#include "CondorStatusUpdater.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <spawn.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <sstream>

using namespace edm::service;

CondorStatusService::CondorStatusService(ParameterSet const& pset, edm::ActivityRegistry& ar)
  :
    m_debug(false),
    m_events(0),
    m_lumis(0),
    m_runs(0),
    m_files(0),
    m_lastUpdate(0)
{
    m_shouldUpdate.clear();
    if (pset.exists("debug"))
    {
        m_debug = true;
    }
    if (!isChirpSupported()) {return;}

    ar.watchPostCloseFile(this, &CondorStatusService::filePost);
    ar.watchPostEvent(this, &CondorStatusService::eventPost);
    ar.watchPostGlobalEndLumi(this, &CondorStatusService::lumiPost);
    ar.watchPostGlobalEndRun(this, &CondorStatusService::runPost);
    ar.watchPostBeginJob(this, &CondorStatusService::beginPost);
    ar.watchPostEndJob(this, &CondorStatusService::endPost);

    if (pset.exists("updateIntervalSeconds"))
    {
        setUpdateInterval(pset.getUntrackedParameter<unsigned int>("updateIntervalSeconds"));
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
CondorStatusService::beginPost()
{
    m_beginJob = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    update();
}


void
CondorStatusService::endPost()
{
    forceUpdate();
}


bool
CondorStatusService::isChirpSupported()
{
    if (m_debug) {return true;}

    return getenv("_CONDOR_CHIRP_CONFIG") && updateChirp("ChirpCMSSWElapsed", "0");
}


void
CondorStatusService::forceUpdate()
{
    time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    updateImpl(now - m_lastUpdate);
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
    std::stringstream ss_elapsed; ss_elapsed << jobTime;
    updateChirp("ChirpCMSSWElapsed", ss_elapsed.str());

    std::stringstream ss_events; ss_events << m_events;
    updateChirp("ChirpCMSSWEvents", ss_events.str());

    std::stringstream ss_lumis; ss_lumis << m_lumis;
    updateChirp("ChirpCMSSWLumis", ss_lumis.str());

    std::stringstream ss_runs; ss_runs << m_runs;
    updateChirp("ChirpCMSSWRuns", ss_runs.str());

    std::stringstream ss_files; ss_files << m_files;
    updateChirp("ChirpCMSSWFiles", ss_files.str());

    float ema_coeff = 1 - std::exp(-static_cast<float>(sinceLastUpdate)/m_emaInterval);
    m_rate = ema_coeff*static_cast<float>(m_events-m_lastEventCount)/static_cast<float>(sinceLastUpdate) + (1.0-ema_coeff)*m_rate;
    m_lastEventCount = m_events;
    std::stringstream ss_rate; ss_rate << m_rate;
    updateChirp("ChirpCMSSWEventRate", ss_rate.str());
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

