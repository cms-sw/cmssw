#ifndef RecoTracker_MkFitCore_standalone_ConfigStandalone_h
#define RecoTracker_MkFitCore_standalone_ConfigStandalone_h

#include "RecoTracker/MkFitCore/interface/Config.h"

#include <string>
#include <map>
#include <vector>

namespace mkfit {

  class TrackerInfo;
  class IterationsInfo;

  void execTrackerInfoCreatorPlugin(const std::string& base, TrackerInfo& ti, IterationsInfo& ii, bool verbose = false);

  //------------------------------------------------------------------------------

  // Enum for input seed options
  enum seedOpts { simSeeds, cmsswSeeds, findSeeds };
  typedef std::map<std::string, std::pair<seedOpts, std::string> > seedOptsMap;

  // Enum for seed cleaning options
  enum cleanOpts { noCleaning, cleanSeedsN2, cleanSeedsPure, cleanSeedsBadLabel };
  typedef std::map<std::string, std::pair<cleanOpts, std::string> > cleanOptsMap;

  // Enum for cmssw matching options
  enum matchOpts { trkParamBased, hitBased, labelBased };
  typedef std::map<std::string, std::pair<matchOpts, std::string> > matchOptsMap;

  //------------------------------------------------------------------------------

  namespace Config {

    extern TrackerInfo TrkInfo;
    extern IterationsInfo ItrInfo;

    extern std::string geomPlugin;

    // default file version
    constexpr int FileVersion = 1;

    // config on main + mkFit
    extern int nTracks;  //defined in Config.cc by default or when reading events from file
    extern int nEvents;
    extern int nItersCMSSW;
    extern bool loopOverFile;
    // XXXXMT: nTracks should be thrown out ... SMatrix and Event allocate some arrays on this
    // which can be wrong for real data or in multi-event environment

    // the following are only used in SMatrix version
    constexpr float nSigma = 3.;
    constexpr float minDPhi = 0.01;  // default: 0.;  cmssw tests: 0.01;
    constexpr float maxDPhi = Const::PI;
    constexpr float minDEta = 0.;
    constexpr float maxDEta = 1.0;

    // Configuration for simulation info
    constexpr int NiterSim = 10;  // Can make more steps due to near volume misses.
    // CMS beam spot width 25um in xy and 5cm in z
    constexpr float beamspotX = 0.1;
    constexpr float beamspotY = 0.1;
    constexpr float beamspotZ = 1.0;

    // XXMT4K minPt was 0.5. Figure out what is the new limit for 90cm or be
    // more flexible about finding fewer hits. Or postprocess looper candidates.
    constexpr float minSimPt = 1;
    constexpr float maxSimPt = 10.;

    // XXMT Hardhack -- transition region excluded in Simulation::setupTrackByToyMC()
    constexpr float minSimEta = -2.4;
    constexpr float maxSimEta = 2.4;
    // For testing separate EC-/BRL/EC+; -2.3--1.5 / -0.9-0.9 / 1.5-2.3
    //constexpr float minSimEta =  -0.9;
    //constexpr float maxSimEta =   0.9;

    constexpr float hitposerrXY = 0.01;  // resolution is 100um in xy --> more realistic scenario is 0.003
    constexpr float hitposerrZ = 0.1;    // resolution is 1mm in z
    constexpr float hitposerrR = Config::hitposerrXY / 10.0f;  // XXMT4K ??? I don't get this ...
    constexpr float varXY = Config::hitposerrXY * Config::hitposerrXY;
    constexpr float varZ = Config::hitposerrZ * Config::hitposerrZ;
    constexpr float varR = Config::hitposerrR * Config::hitposerrR;

    // scattering simulation
    constexpr float X0 =
        9.370;  // cm, from http://pdg.lbl.gov/2014/AtomicNuclearProperties/HTML/silicon_Si.html // Pb = 0.5612 cm
    constexpr float xr =
        0.1;  //  -assumes radial impact. This is bigger than what we have in main --> shouldn't it be the parameter below??? if radial impact??
    //const     float xr = std::sqrt(Config::beamspotX*Config::beamspotX + Config::beamspotY*Config::beamspotY);

    // Config for seeding
    constexpr int nlayers_per_seed_max = 4;  // Needed for allocation of arrays on stack.
    constexpr float chi2seedcut = 9.0;
    constexpr float lay01angdiff =
        0.0634888;  // analytically derived... depends on geometry of detector --> from mathematica ... d0 set to one sigma of getHypot(bsX,bsY)
    constexpr float lay02angdiff = 0.11537;
    constexpr float dEtaSeedTrip =
        0.06;  // for almost max efficiency --> empirically derived... depends on geometry of detector
    constexpr float dPhiSeedTrip =
        0.0458712;  // numerically+semianalytically derived... depends on geometry of detector
    // Recalculated in seedTest as it depends on nlayers_per_seed
    // static const float seed_z2cut= (nlayers_per_seed * fRadialSpacing) / std::tan(2.0f*std::atan(std::exp(-1.0f*dEtaSeedTrip)));
    constexpr float seed_z0cut = beamspotZ * 3.0f;   // 3cm
    constexpr float seed_z1cut = hitposerrZ * 3.6f;  // 3.6 mm --> to match efficiency from chi2cut
    constexpr float seed_d0cut = 0.5f;               // 5mm
    extern bool cf_seeding;

    // config for seeding as well... needed bfield
    constexpr float maxCurvR = (100 * minSimPt) / (Const::sol * Bfield);  // in cm

    // Config for Conformal fitter --> these change depending on inward/outward, which tracks used (MC vs reco), geometry, layers used, track params generated...
    // parameters for layers 0,4,9
    constexpr float blowupfit = 10.0;
    constexpr float ptinverr049 =
        0.0078;  // 0.0075; // errors used for MC only fit, straight from sim tracks, outward with simple geometry
    constexpr float phierr049 = 0.0017;    // 0.0017;
    constexpr float thetaerr049 = 0.0033;  // 0.0031;
    // parameters for layers 0,1,2 // --> ENDTOEND with "real seeding", fit is outward by definition, with poly geo
    constexpr float ptinverr012 = 0.12007;  // 0.1789;  -->old values from only MC seeds
    constexpr float phierr012 = 1.0;        // found empirically 0.00646; // 0.0071
    constexpr float thetaerr012 = 0.2;      // also found empirically 0.01366; // 0.0130;

    // config on fitting
    extern bool cf_fitting;

    extern bool mtvLikeValidation;
    extern bool mtvRequireSeeds;
    // Selection of simtracks from CMSSW. Used in Event::clean_cms_simtracks() and MkBuilder::prep_cmsswtracks()
    extern int cmsSelMinLayers;

    // config on validation
    extern int nMinFoundHits;
    constexpr float minCMSSWMatchChi2[6] = {100, 100, 50, 50, 30, 20};
    constexpr float minCMSSWMatchdPhi[6] = {0.2, 0.2, 0.1, 0.05, 0.01, 0.005};
    extern bool quality_val;
    extern bool sim_val_for_cmssw;
    extern bool sim_val;
    extern bool cmssw_val;
    extern bool fit_val;
    extern bool readSimTrackStates;  // need this to fill pulls
    extern bool inclusiveShorts;
    extern bool keepHitInfo;
    extern bool tryToSaveSimInfo;
    extern matchOpts cmsswMatchingFW;
    extern matchOpts cmsswMatchingBK;

    // config on dead modules
    extern bool useDeadModules;

    // number of layer1 hits for finding seeds per task
    extern int numHitsPerTask;

    // seed options
    extern seedOpts seedInput;
    extern cleanOpts seedCleaning;
    extern bool readCmsswTracks;

    extern bool dumpForPlots;

    extern bool kludgeCmsHitErrors;
    extern bool backwardFit;
    extern bool backwardSearch;

    extern int numThreadsSimulation;
    extern int finderReportBestOutOfN;

    extern bool includePCA;

    // ================================================================

    extern bool silent;
    extern bool json_verbose;
    extern bool json_dump_before;
    extern bool json_dump_after;
    extern std::vector<std::string> json_patch_filenames;
    extern std::vector<std::string> json_load_filenames;
    extern std::string json_save_iters_fname_fmt;
    extern bool json_save_iters_include_iter_info_preamble;

    // ================================================================

    void recalculateDependentConstants();

  }  // end namespace Config

}  // end namespace mkfit

#endif
