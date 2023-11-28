#include "PerfTools/JeProf/interface/jeprof.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <mutex>

extern "C" {
typedef int (*mallctl_t)(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen);
}

namespace {
  bool initialize_prof();
  mallctl_t mallctl = nullptr;
  const bool have_jemalloc_and_prof = initialize_prof();

  bool initialize_prof() {
    // check if mallctl and friends are available, if we are using jemalloc
    mallctl = (mallctl_t)::dlsym(RTLD_DEFAULT, "mallctl");
    if (mallctl == nullptr)
      return false;
    // check if heap profiling available, if --enable-prof was specified at build time
    bool enable_prof = false;
    size_t bool_s = sizeof(bool);
    mallctl("prof.active", &enable_prof, &bool_s, nullptr, 0);
    return enable_prof;
  }
}  // namespace

namespace cms::jeprof {
  std::once_flag warning_flag;
  void makeHeapDump(const char *fileName) {
    std::once_flag warning_flag;
    if (!have_jemalloc_and_prof) {
      std::call_once(warning_flag,
                     []() {
                       edm::LogWarning("JeProfModule")
                           << "JeProfModule requested but application is not"
                           << " currently being profiled with jemalloc profiling enabled\n"
                           << "Enable jemalloc profiling by running\n"
                           << "MALLOC_CONF=prof_leak:true,lg_prof_sample:10,prof_final:true cmsRunJEProf config.py\n";
                     });
      return;
    }
    mallctl("prof.dump", nullptr, nullptr, &fileName, sizeof(const char *));
  }
}  // namespace cms::jeprof
