
/**
 * CondorDynamicResize plugin - control the number of active TBB threads.
 *
 * This plugin will periodically check the HTCondor job environment and tune
 * the number of active TBB streams according to the number of cores allocated
 * by HTCondor.
 *
 * The number of streams is tuned by submitting a TBB task which does nothing
 * but "sleep" until this plugin decides it should be unblocked.
 *
 * Tuning the number of "sleep" TBB tasks is managed by a helper thread.  The
 * helper thread does two things:
 * 1) Listens for the PostEndJob signal; when this is received, signal all
 *    TBB tasks to terminate.
 * 2) Periodically (once a second), wakeup and re-parse HTCondor's "machine ad"
 *    file.  This file contains a description of the resources allocated by
 *    HTCondor.  Stop/start additional 'sleep tasks' so the number of active
 *    threads matches the number of cores allocated by HTCondor.
 *
 * This plugin was written by Brian Bockelman; work was started on 2 May 2017.
 */

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"

#include "tbb/task.h"

#include <atomic>
#include <cassert>
#include <cctype>
#include <chrono>
#include <iostream>
#include <fstream>
#include <future>
#include <sstream>
#include <thread>
#include <vector>

namespace edm {

  namespace service {

    class CondorDynamicResizeService
    {

    public:

      explicit CondorDynamicResizeService(ParameterSet const& pset,
                                          edm::ActivityRegistry& ar);
      ~CondorDynamicResizeService() {}
      CondorDynamicResizeService(const CondorDynamicResizeService&) = delete;
      CondorDynamicResizeService& operator=(const CondorDynamicResizeService&)
        = delete;

      static void fillDescriptions(ConfigurationDescriptions &descriptions);

    private:

      // Main event loop for the management thread.
      void manageTasks(std::future<int> stop_signal);
      static void manageTasksHelper(CondorDynamicResizeService *svc,
                                    std::future<int> stop_signal);

      // Parse the HTCondor environment and determine the optimal number
      // of cores to utilise.  Returns 0 on error.  As m_max_cores can change
      // during the process life, getTargetCores may be larger than max cores.
      unsigned int getTargetCores() const;

      // Signal from CMSSW that we should shutdown all our threads and tasks.
      void shutdown();

      bool m_debug = false;

      // Set of promises used to halt TBB sleeper tasks.
      std::vector<std::promise<int>> m_sleeper_promises;

      // Indicates management thread should halt all futures and shutdown.
      std::promise<int> m_stop_signal;

      // Management of thread counts.  This is an atomic because this will
      // be read from the helper thread (m_mgmt_thread) and written to in
      // a callback invoked by the framework.
      std::atomic<unsigned int> m_max_cores;

      // Management thread
      std::thread m_mgmt_thread;
    };

  }

}

namespace {

  void trim(std::string& s, const std::string& drop = " \t") {
    std::string::size_type p = s.find_last_not_of(drop);
    if (p != std::string::npos) {
          s = s.erase(p+1);
    }
    s = s.erase(0, s.find_first_not_of(drop));
  }


  class SleeperTask : public tbb::task
  {

  public:

    explicit SleeperTask(std::future<int> halt_signal, bool debug) :
      m_debug(debug),
      m_halt_signal(std::move(halt_signal)) {}

    tbb::task* execute() override {
      if (m_debug) {std::cout << "Launching a TBB-based sleep task.\n";}
      m_halt_signal.wait();
      if (m_debug) {std::cout << "Exiting from a TBB-based sleep task.\n";}
      return nullptr;
    }

  private:

    bool m_debug = false;
    std::future<int> m_halt_signal;
  };

}

using namespace edm::service;

CondorDynamicResizeService::CondorDynamicResizeService(
                           ParameterSet const& pset,
                           edm::ActivityRegistry& ar) :
  m_debug(false),
  m_max_cores(1)
{
  if (pset.existsAs<bool>("debug"))
  {
    m_debug = pset.getUntrackedParameter<bool>("debug");
  }

  ar.preallocateSignal_.connect([this](service::SystemBounds const& bounds) {
    if (m_debug) {
      std::cout << "Got preallocate signal "
                << bounds.maxNumberOfThreads() << "\n";
    }
    m_max_cores = bounds.maxNumberOfThreads();
  });

  ar.watchPostEndJob(this, &CondorDynamicResizeService::shutdown);

  m_mgmt_thread = std::thread(manageTasksHelper, this,
                              m_stop_signal.get_future());
}


void
CondorDynamicResizeService::manageTasksHelper(CondorDynamicResizeService *svc,
                  std::future<int> stop_signal)
{
  svc->manageTasks(std::move(stop_signal));
}


void
CondorDynamicResizeService::manageTasks(std::future<int> stop_signal)
{
  if (m_debug) {std::cout << "Task management thread has started!\n";}

  // By default, no cores are blocked.
  unsigned int cores = m_max_cores;
  unsigned int max_cores = m_max_cores;

  // Control of sleeper tasks.
  std::vector<std::promise<int>> sleeper_promises;

  while (1)
  {
    unsigned int loop_max_cores = m_max_cores;

    // When the framework does the preallocate callback, we'll see a one-time jump
    // in max_cores from 1 to N; in this case, we must also assume the number of cores
    // should jump too.  This makes an assumption that, prior to the callback, no TBB
    // threads were blocked.
    if (loop_max_cores > max_cores) {
      max_cores = loop_max_cores;
      cores = max_cores;
    }

    if (m_debug) {
      std::cout << "Querying number of target cores; current running "
                << cores << "\n";
    }
    unsigned int target = getTargetCores();
    target = (max_cores < target) ? max_cores : target;
    if (m_debug) {std::cout << "Resulting target: " << target << "\n";}

    // Target is less than the current number of cores; submit blocker
    // tasks to TBB.
    while (target && (target < cores))
    {
      sleeper_promises.emplace_back();
      tbb::task::enqueue( *new( tbb::task::allocate_root() )
                          SleeperTask(sleeper_promises.back().get_future(),
                                      m_debug) );
      cores --;
    }

    // Target is greater than the current number of cores; remove blocker tasks.
    while (target && (target > cores))
    {
      // If max_cores has increased in the last loop iteration (i.e., the
      // preallocate callback fired), then sleeper_promises might be empty.
      if (!sleeper_promises.empty()) {
        sleeper_promises.back().set_value(1);
        sleeper_promises.pop_back();
      }
      cores ++;
    }

    // Assert the number of active cores is equal to the target cores.
    assert(!target || (cores == target));

    bool is_done = false;
    auto status = stop_signal.wait_for(std::chrono::seconds(1));
    switch (status) {
    case std::future_status::deferred:
      if (m_debug) {
        std::cout << "Logic error -- deferred future shouldn't be possible.\n";
      }
      return;
    case std::future_status::ready:
      if (m_debug) {
        std::cout << "Management thread shutdown requested.\n";
      }
      is_done = true;
      break;
    case std::future_status::timeout:
      if (m_debug) {
        std::cout << "Management thread woke up.\n";
      }
      break;
    }
    if (is_done) {break;}
  }

  // When the sleeper_promises container is destroyed, the associated
  // std::future's appear to be fired on the current architecture.  I
  // couldn't find documentation of this behavior, so I explicitly set the
  // values to be safe.
  for (auto &promise : sleeper_promises)
  {
    promise.set_value(1);
  }
}


unsigned int
CondorDynamicResizeService::getTargetCores() const
{
  const char *machine_ad = getenv("_CONDOR_MACHINE_AD");
  if (!machine_ad) {return 0;}

  if (m_debug) {std::cout << "Parsing machine ad file: " << machine_ad << "\n";}
  std::ifstream fmachine_ad(machine_ad);
  if (!fmachine_ad.is_open()) {
    return 0;
  }

  std::string buf; buf.reserve(1024);
  while (!fmachine_ad.eof()) {
    buf.clear();
    std::getline(fmachine_ad, buf);

    std::istringstream iss(buf);
    std::string token;
    std::string property;
    std::string value;
    int iteration = 0;
    while (std::getline(iss, token, '=')) {
      switch (iteration) {
      case 0:
        property = token;
        break;
      case 1:
        value = token;
      default:
        break;
      }
      iteration ++;
    }
    trim(property);
    std::transform(property.begin(), property.end(),
                   property.begin(),
                   ::tolower);
    trim(value);

    if (property == "cpus") {
      try {
        return std::stoul(value);
      } catch (...) {
        return 0;
      }
    }
  }
  return 0;
}


void
CondorDynamicResizeService::shutdown()
{
  // Tell the management thread it should exit, then wait for that to finish.
  m_stop_signal.set_value(1);
  m_mgmt_thread.join();
}


void
CondorDynamicResizeService::fillDescriptions(
    ConfigurationDescriptions &descriptions)
{
  ParameterSetDescription desc;
  desc.setComment("Service to update number of active threads dynamically, "
                  "based on HTCondor environment");
  desc.addOptionalUntracked<bool>("debug", false)
    ->setComment("Enable debugging of this service.");
  descriptions.add("CondorDynamicResizeService", desc);
}

// Invoke framework macros to actually create / initialize / register this
// service
typedef edm::serviceregistry::AllArgsMaker<
                                     edm::service::CondorDynamicResizeService>
        CondorDynamicResizeServiceMaker;

DEFINE_FWK_SERVICE_MAKER(CondorDynamicResizeService,
                         CondorDynamicResizeServiceMaker);

