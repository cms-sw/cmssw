#ifndef FWCore_Concurrency_StreamNumaDomains_h
#define FWCore_Concurrency_StreamNumaDomains_h
//
// The NUMA domains of the cpus available to the job, and a fixed assignment of the streams to them, in proportion to
// the number of cpus of each domain (contiguous blocks of streams). Enabled by process.options.numaLocality, which
// turns on together:
//   - one TBB arena per domain, whose threads are pinned to the cpus of the domain; each event runs in the arena of its
//     stream's domain;
//   - the placement of the per-stream objects: while a per-stream object is constructed (stream schedule and workers,
//     stream modules, event principal, beginStream), the constructing thread is pinned to the cpus of the stream's
//     domain, so that the memory it first touches lands on that domain;
//   - the replication of the EventSetup data: each domain gets its own copy of the EventSetup products, produced by a
//     task run in the arena of that domain (see runInDomain), and each stream reads the copies of its own domain;
//     process.options.eventSetup.numaReplicatedRecords and numaExcludedRecords restrict the replication to, or
//     exclude from it, some records.
//
// Nothing changes if numaLocality is not enabled, or if the job's cpus are all in one NUMA domain.
//

#include <sched.h>

#include <atomic>
#include <string>
#include <string_view>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

namespace edm::numa {

  class StreamNumaDomains {
  public:
    static StreamNumaDomains& instance();

    struct Configuration {
      bool numaLocality = false;
      std::vector<std::string> replicatedRecords;
      std::vector<std::string> excludedRecords;
    };

    // true if the job's affinity mask is smaller than the set of cpus it could use, i.e. the effective cpuset of its
    // cgroup (obtained from the kernel, see the implementation). A job pinned to all of them is not restricted.
    static bool affinityRestricted();

    // number of cpus in the job's affinity mask
    unsigned int numberOfJobCpus() const { return CPU_COUNT(&all_); }

    // compute the domains and the assignment of the streams; called by each EventProcessor, before any per-stream
    // object is created
    void setup(unsigned int nStreams, Configuration const& config);

    // forget the domains, the assignment of the streams and the arenas; called by the EventProcessor's destructor, so
    // that a later EventProcessor in the same process (e.g. in a unit test) does not inherit them
    void reset();

    // true if numaLocality is enabled and the job's cpus span more than one NUMA domain (the domains are computed only
    // with numaLocality): the per-domain arenas, the placement of the per-stream objects and the EventSetup replication
    // are all used
    bool enabled() const { return domains_.size() > 1; }

    // true if the products of the EventSetup record should be replicated on each domain
    bool esReplicateRecord(std::string_view record) const;

    unsigned int size() const { return domains_.size(); }
    int node(unsigned int domain) const { return domains_[domain].node; }
    unsigned int numberOfCpus(unsigned int domain) const { return domains_[domain].count; }
    cpu_set_t const& cpus(unsigned int domain) const { return domains_[domain].cpus; }
    cpu_set_t const& allCpus() const { return all_; }
    unsigned int domainOfStream(unsigned int stream) const { return streamDomain_[stream]; }
    // the domain whose EventSetup copies are used by a stream (0 without EventSetup replication)
    unsigned int esDomainOfStream(unsigned int stream) const { return enabled() ? streamDomain_[stream] : 0; }

    // the per-domain TBB arenas, registered by the EventProcessor; nullptr if there are none
    void setArena(unsigned int domain, oneapi::tbb::task_arena* arena);
    oneapi::tbb::task_arena* arena(int domain) const {
      bool valid = domain >= 0 and static_cast<unsigned int>(domain) < domainArenas_.size();
      return valid ? domainArenas_[domain] : nullptr;
    }

  private:
    StreamNumaDomains() = default;

    struct Domain {
      int node;
      unsigned int count;
      cpu_set_t cpus;
    };
    // the job's cpus, split by NUMA node (libnuma)
    static std::vector<Domain> findDomains(cpu_set_t& all);

    std::vector<Domain> domains_;
    std::vector<unsigned int> streamDomain_;
    cpu_set_t all_;
    std::vector<std::string> esRecords_;
    std::vector<std::string> esExcluded_;
    std::vector<oneapi::tbb::task_arena*> domainArenas_;
  };

  // the domain of the per-domain arena the calling thread is running in, or -1; kept up to date by the arena observers,
  // which call enterArena() and leaveArena().
  // This assumes that the per-domain arenas are never nested, i.e. that a thread never enters one of them (with
  // task_arena::execute) while it is already running in another one: leaving the inner arena would then set the
  // domain to -1 instead of the outer one, and the arena observers would set the thread's affinity to all the cpus of
  // the job instead of those of the outer domain. The framework only ever enqueues work into another domain's arena
  // (see runInDomain and EventProcessor::processEventAsync).
  int currentArenaDomain();
  void enterArena(unsigned int domain);
  void leaveArena();

  // counters, reported at the end of the job by reportArenaStatistics()
  struct ArenaStatistics {
    std::atomic<unsigned long> entries{0};     // threads entering a per-domain arena
    std::atomic<unsigned long> moves{0};       // ... coming from a different per-domain arena than their previous one
    std::atomic<unsigned long> dispatched{0};  // tasks enqueued into the arena of their domain by runInDomain
  };
  ArenaStatistics& arenaStatistics();
  std::string reportArenaStatistics();

  // Run f as part of group, in the arena of the given NUMA domain. If domain is negative, there is no arena for it, or
  // the calling thread already runs in that arena, this is group.run(f).
  template <typename F>
  void runInDomain(oneapi::tbb::task_group& group, int domain, F&& f) {
    if (domain >= 0 and domain != currentArenaDomain()) {
      if (auto* arena = StreamNumaDomains::instance().arena(domain)) {
        ++arenaStatistics().dispatched;
        arena->enqueue(std::forward<F>(f), group);
        return;
      }
    }
    group.run(std::forward<F>(f));
  }

  // Keep the calling thread pinned to the cpus of the NUMA domain of a stream (only with numaLocality), so that the
  // memory it first touches lands on that domain. moveToStream() pins the thread to the domain of the given stream,
  // and does nothing if the thread is already pinned to it, by this object or by an enclosing one; the affinity the
  // thread had before the first change is restored when the object goes out of scope. One object around a serial
  // loop over the streams pins the thread once per domain rather than once per stream, since the streams of a domain
  // are contiguous; one object per task does the same when the per-stream work runs in parallel tasks.
  class StreamPlacement {
  public:
    StreamPlacement() = default;
    ~StreamPlacement();

    StreamPlacement(StreamPlacement const&) = delete;
    StreamPlacement& operator=(StreamPlacement const&) = delete;

    void moveToStream(unsigned int stream);

  private:
    cpu_set_t previous_;
    int previousDomain_ = -1;
    bool active_ = false;
  };

}  // namespace edm::numa

#endif
