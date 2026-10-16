#include "FWCore/Concurrency/interface/StreamNumaDomains.h"

#include <dlfcn.h>
#include <numa.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

namespace {
  // With jemalloc, flush the thread cache of the calling thread: after a thread moves to another NUMA domain, its cache
  // would otherwise keep serving small allocations with memory from the arena (and node) it used before.
  void flushThreadCache() {
    using mallctl_t = int (*)(char const*, void*, size_t*, void*, size_t);
    static mallctl_t const mallctl = reinterpret_cast<mallctl_t>(dlsym(RTLD_DEFAULT, "mallctl"));
    if (mallctl != nullptr) {
      mallctl("thread.tcache.flush", nullptr, nullptr, nullptr, 0);
    }
  }

  // the NUMA domain the calling thread is currently pinned to by a StreamPlacement, or -1
  thread_local int currentDomain = -1;

  // the per-domain arena the calling thread is running in (-1 if none), and the last one it was in
  thread_local int currentArena = -1;
  thread_local int previousArena = -1;
}  // namespace

namespace edm::numa {

  StreamNumaDomains& StreamNumaDomains::instance() {
    static StreamNumaDomains domains;
    return domains;
  }

  void StreamNumaDomains::setArena(unsigned int domain, oneapi::tbb::task_arena* arena) {
    if (domainArenas_.size() <= domain) {
      domainArenas_.resize(domain + 1, nullptr);
    }
    domainArenas_[domain] = arena;
  }

  int currentArenaDomain() { return currentArena; }

  void enterArena(unsigned int domain) {
    auto& stats = arenaStatistics();
    ++stats.entries;
    // a thread entering the arena of a domain from another arena, or for the first time, may still hold memory from
    // another node in its jemalloc cache: flush it
    if (previousArena != static_cast<int>(domain)) {
      if (previousArena >= 0) {
        ++stats.moves;
      }
      flushThreadCache();
    }
    currentArena = domain;
    previousArena = domain;
  }

  void leaveArena() { currentArena = -1; }

  ArenaStatistics& arenaStatistics() {
    static ArenaStatistics stats;
    return stats;
  }

  std::string reportArenaStatistics() {
    auto const& stats = arenaStatistics();
    unsigned long entries = stats.entries;
    unsigned long moves = stats.moves;
    unsigned long dispatched = stats.dispatched;
    std::ostringstream out;
    out << "NUMA arena statistics:\n";
    out << "  " << entries << " times a thread entered the arena of a NUMA domain;\n";
    out << "  " << moves << " of them";
    if (entries > 0) {
      out << " (" << std::fixed << std::setprecision(1) << 100. * moves / entries << "%)";
    }
    out << " came from the arena of a different domain: the thread moved to the cpus of another NUMA node, where its\n"
        << "    caches are cold and the data it touched before are remote (its jemalloc thread cache is flushed);\n";
    out << "  " << dispatched << " tasks were sent to the arena of their domain from a thread running outside of it\n"
        << "    (e.g. the production of the EventSetup copies of a domain, or the tasks queued for it).";
    return out.str();
  }

  std::vector<StreamNumaDomains::Domain> StreamNumaDomains::findDomains(cpu_set_t& all) {
    std::vector<Domain> domains;
    CPU_ZERO(&all);
    if (numa_available() < 0 or sched_getaffinity(0, sizeof(all), &all) != 0) {
      return domains;
    }
    struct bitmask* nodeCpus = numa_allocate_cpumask();
    for (int node = 0; node <= numa_max_node(); ++node) {
      if (numa_node_to_cpus(node, nodeCpus) != 0) {
        continue;
      }
      Domain domain{node, 0, {}};
      CPU_ZERO(&domain.cpus);
      for (unsigned int cpu = 0; cpu < nodeCpus->size and cpu < CPU_SETSIZE; ++cpu) {
        if (numa_bitmask_isbitset(nodeCpus, cpu) and CPU_ISSET(cpu, &all)) {
          CPU_SET(cpu, &domain.cpus);
          ++domain.count;
        }
      }
      if (domain.count > 0) {
        domains.push_back(domain);
      }
    }
    numa_free_cpumask(nodeCpus);
    return domains;
  }

  bool StreamNumaDomains::affinityRestricted() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
      return false;
    }
    // The cpus the job could use are those of the effective cpuset of its cgroup. Rather than reading the cgroup files,
    // which depends on the cgroup version (v1 or v2), on whether and where cgroupfs is mounted inside a container, and
    // on cgroup namespaces, ask the kernel directly: sched_setaffinity() clips any request to the cpuset, so requesting
    // all the cpus and reading back the granted ones gives the cpuset, whatever the environment.
    //
    // The probe runs in a throwaway thread, so that the affinity of the job's threads is never changed. Widening the
    // affinity of the calling thread and restoring it afterwards would not be equivalent: the restore can fail (e.g. if
    // the cpuset shrinks or cpus go offline in between, or out of memory), leaving the thread on the wide mask; and
    // sched_getaffinity() only returns the effective mask, without the offline cpus and the cpus outside the cpuset,
    // while the kernel (since 6.2) remembers the mask the thread asked for and re-applies it when the cpuset changes or
    // cpus come back online. Restoring the effective mask would replace the requested one, for the calling thread and
    // for the threads it creates afterwards. A throwaway thread inherits both masks, changes only its own, and exits.
    cpu_set_t available;
    CPU_ZERO(&available);
    bool known = false;
    std::thread([&available, &known]() {
      cpu_set_t all;
      CPU_ZERO(&all);
      for (unsigned int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        CPU_SET(cpu, &all);
      }
      known = sched_setaffinity(0, sizeof(all), &all) == 0 and sched_getaffinity(0, sizeof(available), &available) == 0;
    }).join();
    if (not known) {
      return false;
    }
    // restricted if the job's mask is a strict subset of the cpus it could use
    cpu_set_t common;
    CPU_AND(&common, &mask, &available);
    return CPU_EQUAL(&common, &mask) and not CPU_EQUAL(&mask, &available);
  }

  void StreamNumaDomains::reset() {
    domains_.clear();
    streamDomain_.clear();
    CPU_ZERO(&all_);
    esRecords_.clear();
    esExcluded_.clear();
    domainArenas_.clear();
    auto& stats = arenaStatistics();
    stats.entries = 0;
    stats.moves = 0;
    stats.dispatched = 0;
  }

  void StreamNumaDomains::setup(unsigned int nStreams, Configuration const& config) {
    reset();
    esRecords_ = config.replicatedRecords;
    esExcluded_ = config.excludedRecords;
    if (not config.numaLocality) {
      return;
    }

    // the cpus available to the job, split by NUMA node
    domains_ = findDomains(all_);
    unsigned int total = 0;
    for (auto const& domain : domains_) {
      total += domain.count;
    }

    // assign the streams to the domains in proportion to their cpus, in contiguous blocks
    streamDomain_.assign(nStreams, 0);
    if (domains_.size() < 2) {
      return;
    }
    unsigned int domain = 0;
    unsigned int boundary = domains_[0].count;
    for (unsigned int stream = 0; stream < nStreams; ++stream) {
      while (domain + 1 < domains_.size() and stream * total >= boundary * nStreams) {
        ++domain;
        boundary += domains_[domain].count;
      }
      streamDomain_[stream] = domain;
    }
  }

  bool StreamNumaDomains::esReplicateRecord(std::string_view record) const {
    if (not enabled()) {
      return false;
    }
    if (not esRecords_.empty() and std::find(esRecords_.begin(), esRecords_.end(), record) == esRecords_.end()) {
      return false;
    }
    return std::find(esExcluded_.begin(), esExcluded_.end(), record) == esExcluded_.end();
  }

  void StreamPlacement::moveToStream(unsigned int stream) {
    auto const& domains = StreamNumaDomains::instance();
    if (not domains.enabled()) {
      return;
    }
    int domain = domains.domainOfStream(stream);
    if (domain == currentDomain) {
      return;  // already pinned to the right domain, by this object or by an enclosing one
    }
    if (not active_) {
      // first change: remember the affinity to restore at the end of the scope
      if (sched_getaffinity(0, sizeof(previous_), &previous_) != 0) {
        return;
      }
      previousDomain_ = currentDomain;
    }
    if (sched_setaffinity(0, sizeof(cpu_set_t), &domains.cpus(domain)) != 0) {
      return;
    }
    flushThreadCache();
    currentDomain = domain;
    active_ = true;
  }

  StreamPlacement::~StreamPlacement() {
    if (active_) {
      sched_setaffinity(0, sizeof(previous_), &previous_);
      flushThreadCache();
      currentDomain = previousDomain_;
    }
  }

}  // namespace edm::numa
