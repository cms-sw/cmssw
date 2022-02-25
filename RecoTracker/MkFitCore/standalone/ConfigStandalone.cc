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
    std::string soname = base + ".so";

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
      path += soname;
      if (stat(path.c_str(), &st) == 0) {
        printf("mkfit::execTrackerInfoCreatorPlugin processing '%s'\n", path.c_str());

        void *h = dlopen(path.c_str(), RTLD_LAZY);
        if (!h) {
          perror("dlopen failed");
          exit(2);
        }

        long long *p2f = (long long *)dlsym(h, "TrackerInfoCrator_ptr");
        if (!p2f) {
          perror("dlsym failed");
          exit(2);
        }

        TrackerInfoCreator_foo foo = (TrackerInfoCreator_foo)(*p2f);
        foo(ti, ii, verbose);

        return;
      }

      ++si;
    }

    fprintf(stderr, "TrackerInfo plugin '%s' not found in search path.\n", soname.c_str());
    exit(2);
  }

}  // namespace mkfit
