#include "RecoTracker/MkFitCore/standalone/ConfigStandalone.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

// For plugin loader
#include <dlfcn.h>
#include <sys/stat.h>
#include <cstdlib>

namespace mkfit {

  namespace Config {

    TrackerInfo TrkInfo;
    IterationsInfo ItrInfo;

    std::string geomPlugin = "CylCowWLids";

    int nTracks = 10000;
    int nEvents = 20;
    int nItersCMSSW = 0;
    bool loopOverFile = false;

    seedOpts seedInput = simSeeds;
    cleanOpts seedCleaning = noCleaning;

    bool readCmsswTracks = false;

    bool dumpForPlots = false;

    bool cf_seeding = false;
    bool cf_fitting = false;

    bool quality_val = false;
    bool sim_val_for_cmssw = false;
    bool sim_val = false;
    bool cmssw_val = false;
    bool fit_val = false;
    bool readSimTrackStates = false;
    bool inclusiveShorts = false;
    bool keepHitInfo = false;
    bool tryToSaveSimInfo = false;
    matchOpts cmsswMatchingFW = hitBased;
    matchOpts cmsswMatchingBK = trkParamBased;

    bool useDeadModules = false;

    // number of hits per task for finding seeds
    int numHitsPerTask = 32;

    bool mtvLikeValidation = false;
    bool mtvRequireSeeds = false;
    int cmsSelMinLayers = 12;
    int nMinFoundHits = 10;

    bool kludgeCmsHitErrors = false;
    bool backwardFit = false;
    bool backwardSearch = true;

    int numThreadsSimulation = 12;

    int finderReportBestOutOfN = 1;

    bool includePCA = false;

    // ================================================================

    bool silent = false;
    bool json_verbose = false;
    bool json_dump_before = false;
    bool json_dump_after = false;
    std::vector<std::string> json_patch_filenames;
    std::vector<std::string> json_load_filenames;
    std::string json_save_iters_fname_fmt;
    bool json_save_iters_include_iter_info_preamble = false;

    // ================================================================

    void recalculateDependentConstants() {}

  }  // namespace Config

  //==============================================================================
  // Geometry / Configuration Plugin Loader
  //==============================================================================

  namespace {
    const char *search_path[] = {"", "../Geoms/", "Geoms/", "../", nullptr};
    typedef void (*TrackerInfoCreator_foo)(TrackerInfo &, IterationsInfo &, bool verbose);
  }  // namespace

  void execTrackerInfoCreatorPlugin(const std::string &base, TrackerInfo &ti, IterationsInfo &ii, bool verbose) {
    const std::string soname = base + ".so";
    const std::string binname = base + ".bin";
    struct stat st;

    int si = 0;
    while (search_path[si]) {
      std::string path;
      const char *envpath = std::getenv("MKFIT_BASE");
      if (envpath != nullptr) {
        path += envpath;
        path += "/";
      }
      path += search_path[si];
      std::string sopath = path + soname;
      if (stat(sopath.c_str(), &st) == 0) {
        printf("execTrackerInfoCreatorPlugin processing '%s'\n", sopath.c_str());

        void *h = dlopen(sopath.c_str(), RTLD_LAZY);
        if (!h) {
          perror("dlopen failed");
          fprintf(stderr, "dlerror:\n%s\n", dlerror());
          exit(2);
        }

        long long *p2f = (long long *)dlsym(h, "TrackerInfoCreator_ptr");
        if (!p2f) {
          perror("dlsym failed");
          exit(2);
        }

        std::string binpath = path + binname;
        int binsr = stat(binpath.c_str(), &st);
        printf("execTrackerInfoCreatorPlugin has%s found TrackerInfo binary file '%s'\n",
               binsr ? " NOT" : "",
               binpath.c_str());
        if (binsr == 0)
          ti.read_bin_file(binpath);

        TrackerInfoCreator_foo foo = (TrackerInfoCreator_foo)(*p2f);
        foo(ti, ii, verbose);

        // level 2: print shapes and modules, precision 8
        // ti.print_tracker(2, 8);

        return;
      }

      ++si;
    }

    fprintf(stderr, "TrackerInfo plugin '%s' not found in search path.\n", soname.c_str());
    exit(2);
  }

  namespace internal {
    std::vector<DeadVec> deadvectors;
  }

}  // namespace mkfit
