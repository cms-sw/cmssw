#include "TFile.h"
#include "TTree.h"

#include <iostream>
#include <list>
#include <unordered_map>
#include "RecoTracker/MkFitCore/standalone/Event.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCMS/interface/LayerNumberConverter.h"

using namespace mkfit;

using TrackAlgorithm = TrackBase::TrackAlgorithm;

constexpr bool useMatched = false;

constexpr int cleanSimTrack_minSimHits = 3;
constexpr int cleanSimTrack_minRecHits = 2;

//check if this is the same as in the release
enum class HitType { Pixel = 0, Strip = 1, Glued = 2, Invalid = 3, Phase2OT = 4, Unknown = 99 };

typedef std::list<std::string> lStr_t;
typedef lStr_t::iterator lStr_i;
void next_arg_or_die(lStr_t& args, lStr_i& i) {
  lStr_i j = i;
  if (++j == args.end() || ((*j)[0] == '-')) {
    std::cerr << "Error: option " << *i << " requires an argument.\n";
    exit(1);
  }
  i = j;
}

bool next_arg_option(lStr_t& args, lStr_i& i) {
  lStr_i j = i;
  if (++j == args.end() || ((*j)[0] == '-')) {
    return false;
  }
  i = j;
  return true;
}

void printHelp(const char* av0) {
  printf(
      "Usage: %s [options]\n"
      "Options:\n"
      "  --help                    print help and exit\n"
      "  --input          <str>    input file\n"
      "  --output         <str>    output file\n"
      "  --geo            <file>   binary TrackerInfo geometry (def: CMS-phase1.bin)\n"
      "  --verbosity      <num>    print details (0 quiet, 1 print counts, 2 print all; def: 0)\n"
      "  --maxevt         <num>    maxevt events to write (-1 for everything in the file def: -1)\n"
      "  --clean-sim-tracks        apply sim track cleaning (def: no cleaning)\n"
      "  --write-all-events        write all events (def: skip events with 0 simtracks or seeds)\n"
      "  --write-rec-tracks        write rec tracks (def: not written)\n"
      "  --apply-ccc               apply cluster charge cut to strip hits (def: false)\n"
      "  --all-seeds               write all seeds from the input file, not only initialStep and hltIter0 (def: "
      "false)\n",
      av0);
}

int main(int argc, char* argv[]) {
  bool haveInput = false;
  std::string inputFileName;
  bool haveOutput = false;
  std::string outputFileName;
  std::string geoFileName("CMS-phase1.bin");

  bool cleanSimTracks = false;
  bool writeAllEvents = false;
  bool writeRecTracks = false;
  bool writeHitIterMasks = false;
  bool applyCCC = false;
  bool allSeeds = false;

  int verbosity = 0;
  long long maxevt = -1;

  int cutValueCCC = 1620;  //Nominal value (from first iteration of CMSSW) is 1620

  lStr_t mArgs;
  for (int i = 1; i < argc; ++i) {
    mArgs.push_back(argv[i]);
  }

  lStr_i i = mArgs.begin();
  while (i != mArgs.end()) {
    lStr_i start = i;

    if (*i == "-h" || *i == "-help" || *i == "--help") {
      printHelp(argv[0]);
      exit(0);
    } else if (*i == "--input") {
      next_arg_or_die(mArgs, i);
      inputFileName = *i;
      haveInput = true;
    } else if (*i == "--output") {
      next_arg_or_die(mArgs, i);
      outputFileName = *i;
      haveOutput = true;
    } else if (*i == "--geo") {
      next_arg_or_die(mArgs, i);
      geoFileName = *i;
    } else if (*i == "--verbosity") {
      next_arg_or_die(mArgs, i);
      verbosity = std::atoi(i->c_str());
    } else if (*i == "--maxevt") {
      next_arg_or_die(mArgs, i);
      maxevt = std::atoi(i->c_str());
    } else if (*i == "--clean-sim-tracks") {
      cleanSimTracks = true;
    } else if (*i == "--write-all-events") {
      writeAllEvents = true;
    } else if (*i == "--write-rec-tracks") {
      writeRecTracks = true;
    } else if (*i == "--write-hit-iter-masks") {
      writeHitIterMasks = true;
    } else if (*i == "--apply-ccc") {
      applyCCC = true;
      if (next_arg_option(mArgs, i)) {
        cutValueCCC = std::atoi(i->c_str());
      }
    } else if (*i == "--all-seeds") {
      allSeeds = true;
    } else {
      fprintf(stderr, "Error: Unknown option/argument '%s'.\n", i->c_str());
      printHelp(argv[0]);
      exit(1);
    }
    mArgs.erase(start, ++i);
  }  //while arguments

  if (not haveOutput or not haveInput) {
    fprintf(stderr, "Error: both input and output are required\n");
    printHelp(argv[0]);
    exit(1);
  }

  using namespace std;

  TFile* f = TFile::Open(inputFileName.c_str());
  if (f == 0) {
    fprintf(stderr, "Failed opening input root file '%s'\n", inputFileName.c_str());
    exit(1);
  }

  TTree* t = (TTree*)f->Get("trackingNtuple/tree");

  bool hasPh2hits = t->GetBranch("ph2_isLower") != nullptr;
  TrackerInfo tkinfo;
  tkinfo.read_bin_file(geoFileName);
  LayerNumberConverter lnc(hasPh2hits ? TkLayout::phase2 : TkLayout::phase1);
  const unsigned int nTotalLayers = lnc.nLayers();
  assert(nTotalLayers == (unsigned int)tkinfo.n_layers());

  int nstot = 0;
  std::vector<int> nhitstot(nTotalLayers, 0);

  unsigned long long event;
  t->SetBranchAddress("event", &event);

  //sim tracks
  std::vector<float>* sim_eta = 0;
  std::vector<float>* sim_px = 0;
  std::vector<float>* sim_py = 0;
  std::vector<float>* sim_pz = 0;
  std::vector<int>* sim_parentVtxIdx = 0;
  std::vector<int>* sim_q = 0;
  std::vector<int>* sim_event = 0;
  std::vector<int>* sim_bunchCrossing = 0;
  std::vector<int>* sim_nValid = 0;  //simHit count, actually
  t->SetBranchAddress("sim_eta", &sim_eta);
  t->SetBranchAddress("sim_px", &sim_px);
  t->SetBranchAddress("sim_py", &sim_py);
  t->SetBranchAddress("sim_pz", &sim_pz);
  t->SetBranchAddress("sim_parentVtxIdx", &sim_parentVtxIdx);
  t->SetBranchAddress("sim_q", &sim_q);
  t->SetBranchAddress("sim_event", &sim_event);
  t->SetBranchAddress("sim_bunchCrossing", &sim_bunchCrossing);
  t->SetBranchAddress("sim_nValid", &sim_nValid);

  std::vector<vector<int>>* sim_trkIdx = 0;
  t->SetBranchAddress("sim_trkIdx", &sim_trkIdx);

  //simvtx
  std::vector<float>* simvtx_x = 0;
  std::vector<float>* simvtx_y = 0;
  std::vector<float>* simvtx_z = 0;
  t->SetBranchAddress("simvtx_x", &simvtx_x);
  t->SetBranchAddress("simvtx_y", &simvtx_y);
  t->SetBranchAddress("simvtx_z", &simvtx_z);

  //simhit
  std::vector<short>* simhit_process = 0;
  std::vector<int>* simhit_particle = 0;
  std::vector<int>* simhit_simTrkIdx = 0;
  std::vector<float>* simhit_x = 0;
  std::vector<float>* simhit_y = 0;
  std::vector<float>* simhit_z = 0;
  std::vector<float>* simhit_px = 0;
  std::vector<float>* simhit_py = 0;
  std::vector<float>* simhit_pz = 0;
  t->SetBranchAddress("simhit_process", &simhit_process);
  t->SetBranchAddress("simhit_particle", &simhit_particle);
  t->SetBranchAddress("simhit_simTrkIdx", &simhit_simTrkIdx);
  t->SetBranchAddress("simhit_x", &simhit_x);
  t->SetBranchAddress("simhit_y", &simhit_y);
  t->SetBranchAddress("simhit_z", &simhit_z);
  t->SetBranchAddress("simhit_px", &simhit_px);
  t->SetBranchAddress("simhit_py", &simhit_py);
  t->SetBranchAddress("simhit_pz", &simhit_pz);

  std::vector<std::vector<int>>* simhit_hitIdx = 0;
  t->SetBranchAddress("simhit_hitIdx", &simhit_hitIdx);
  std::vector<std::vector<int>>* simhit_hitType = 0;
  t->SetBranchAddress("simhit_hitType", &simhit_hitType);

  //rec tracks
  std::vector<int>* trk_q = 0;
  std::vector<unsigned int>* trk_nValid = 0;
  std::vector<int>* trk_seedIdx = 0;
  std::vector<unsigned long long>* trk_algoMask = 0;
  std::vector<unsigned int>* trk_algo = 0;
  std::vector<unsigned int>* trk_originalAlgo = 0;
  std::vector<float>* trk_nChi2 = 0;
  std::vector<float>* trk_px = 0;
  std::vector<float>* trk_py = 0;
  std::vector<float>* trk_pz = 0;
  std::vector<float>* trk_pt = 0;
  std::vector<float>* trk_phi = 0;
  std::vector<float>* trk_lambda = 0;
  std::vector<float>* trk_refpoint_x = 0;
  std::vector<float>* trk_refpoint_y = 0;
  std::vector<float>* trk_refpoint_z = 0;
  std::vector<float>* trk_dxyErr = 0;
  std::vector<float>* trk_dzErr = 0;
  std::vector<float>* trk_ptErr = 0;
  std::vector<float>* trk_phiErr = 0;
  std::vector<float>* trk_lambdaErr = 0;
  t->SetBranchAddress("trk_q", &trk_q);
  t->SetBranchAddress("trk_nValid", &trk_nValid);
  t->SetBranchAddress("trk_seedIdx", &trk_seedIdx);
  t->SetBranchAddress("trk_algoMask", &trk_algoMask);
  t->SetBranchAddress("trk_algo", &trk_algo);
  t->SetBranchAddress("trk_originalAlgo", &trk_originalAlgo);
  t->SetBranchAddress("trk_nChi2", &trk_nChi2);
  t->SetBranchAddress("trk_px", &trk_px);
  t->SetBranchAddress("trk_py", &trk_py);
  t->SetBranchAddress("trk_pz", &trk_pz);
  t->SetBranchAddress("trk_pt", &trk_pt);
  t->SetBranchAddress("trk_phi", &trk_phi);
  t->SetBranchAddress("trk_lambda", &trk_lambda);
  t->SetBranchAddress("trk_refpoint_x", &trk_refpoint_x);
  t->SetBranchAddress("trk_refpoint_y", &trk_refpoint_y);
  t->SetBranchAddress("trk_refpoint_z", &trk_refpoint_z);
  t->SetBranchAddress("trk_dxyErr", &trk_dxyErr);
  t->SetBranchAddress("trk_dzErr", &trk_dzErr);
  t->SetBranchAddress("trk_ptErr", &trk_ptErr);
  t->SetBranchAddress("trk_phiErr", &trk_phiErr);
  t->SetBranchAddress("trk_lambdaErr", &trk_lambdaErr);

  std::vector<std::vector<int>>* trk_hitIdx = 0;
  t->SetBranchAddress("trk_hitIdx", &trk_hitIdx);
  std::vector<std::vector<int>>* trk_hitType = 0;
  t->SetBranchAddress("trk_hitType", &trk_hitType);

  //seeds
  std::vector<float>* see_stateTrajGlbX = 0;
  std::vector<float>* see_stateTrajGlbY = 0;
  std::vector<float>* see_stateTrajGlbZ = 0;
  std::vector<float>* see_stateTrajGlbPx = 0;
  std::vector<float>* see_stateTrajGlbPy = 0;
  std::vector<float>* see_stateTrajGlbPz = 0;
  std::vector<float>* see_eta = 0;  //PCA parameters
  std::vector<float>* see_pt = 0;   //PCA parameters
  std::vector<float>* see_stateCcov00 = 0;
  std::vector<float>* see_stateCcov01 = 0;
  std::vector<float>* see_stateCcov02 = 0;
  std::vector<float>* see_stateCcov03 = 0;
  std::vector<float>* see_stateCcov04 = 0;
  std::vector<float>* see_stateCcov05 = 0;
  std::vector<float>* see_stateCcov11 = 0;
  std::vector<float>* see_stateCcov12 = 0;
  std::vector<float>* see_stateCcov13 = 0;
  std::vector<float>* see_stateCcov14 = 0;
  std::vector<float>* see_stateCcov15 = 0;
  std::vector<float>* see_stateCcov22 = 0;
  std::vector<float>* see_stateCcov23 = 0;
  std::vector<float>* see_stateCcov24 = 0;
  std::vector<float>* see_stateCcov25 = 0;
  std::vector<float>* see_stateCcov33 = 0;
  std::vector<float>* see_stateCcov34 = 0;
  std::vector<float>* see_stateCcov35 = 0;
  std::vector<float>* see_stateCcov44 = 0;
  std::vector<float>* see_stateCcov45 = 0;
  std::vector<float>* see_stateCcov55 = 0;
  std::vector<std::vector<float>>* see_stateCurvCov = 0;
  std::vector<int>* see_q = 0;
  std::vector<unsigned int>* see_algo = 0;
  t->SetBranchAddress("see_stateTrajGlbX", &see_stateTrajGlbX);
  t->SetBranchAddress("see_stateTrajGlbY", &see_stateTrajGlbY);
  t->SetBranchAddress("see_stateTrajGlbZ", &see_stateTrajGlbZ);
  t->SetBranchAddress("see_stateTrajGlbPx", &see_stateTrajGlbPx);
  t->SetBranchAddress("see_stateTrajGlbPy", &see_stateTrajGlbPy);
  t->SetBranchAddress("see_stateTrajGlbPz", &see_stateTrajGlbPz);
  t->SetBranchAddress("see_eta", &see_eta);
  t->SetBranchAddress("see_pt", &see_pt);

  bool hasCartCov = t->GetBranch("see_stateCcov00") != nullptr;
  if (hasCartCov) {
    t->SetBranchAddress("see_stateCcov00", &see_stateCcov00);
    t->SetBranchAddress("see_stateCcov01", &see_stateCcov01);
    t->SetBranchAddress("see_stateCcov02", &see_stateCcov02);
    t->SetBranchAddress("see_stateCcov03", &see_stateCcov03);
    t->SetBranchAddress("see_stateCcov04", &see_stateCcov04);
    t->SetBranchAddress("see_stateCcov05", &see_stateCcov05);
    t->SetBranchAddress("see_stateCcov11", &see_stateCcov11);
    t->SetBranchAddress("see_stateCcov12", &see_stateCcov12);
    t->SetBranchAddress("see_stateCcov13", &see_stateCcov13);
    t->SetBranchAddress("see_stateCcov14", &see_stateCcov14);
    t->SetBranchAddress("see_stateCcov15", &see_stateCcov15);
    t->SetBranchAddress("see_stateCcov22", &see_stateCcov22);
    t->SetBranchAddress("see_stateCcov23", &see_stateCcov23);
    t->SetBranchAddress("see_stateCcov24", &see_stateCcov24);
    t->SetBranchAddress("see_stateCcov25", &see_stateCcov25);
    t->SetBranchAddress("see_stateCcov33", &see_stateCcov33);
    t->SetBranchAddress("see_stateCcov34", &see_stateCcov34);
    t->SetBranchAddress("see_stateCcov35", &see_stateCcov35);
    t->SetBranchAddress("see_stateCcov44", &see_stateCcov44);
    t->SetBranchAddress("see_stateCcov45", &see_stateCcov45);
    t->SetBranchAddress("see_stateCcov55", &see_stateCcov55);
  } else {
    t->SetBranchAddress("see_stateCurvCov", &see_stateCurvCov);
  }
  t->SetBranchAddress("see_q", &see_q);
  t->SetBranchAddress("see_algo", &see_algo);

  std::vector<std::vector<int>>* see_hitIdx = 0;
  t->SetBranchAddress("see_hitIdx", &see_hitIdx);
  std::vector<std::vector<int>>* see_hitType = 0;
  t->SetBranchAddress("see_hitType", &see_hitType);

  //pixel hits
  vector<unsigned short>* pix_det = 0;
  vector<unsigned short>* pix_lay = 0;
  vector<unsigned int>* pix_detId = 0;
  vector<float>* pix_x = 0;
  vector<float>* pix_y = 0;
  vector<float>* pix_z = 0;
  vector<float>* pix_xx = 0;
  vector<float>* pix_xy = 0;
  vector<float>* pix_yy = 0;
  vector<float>* pix_yz = 0;
  vector<float>* pix_zz = 0;
  vector<float>* pix_zx = 0;
  vector<int>* pix_csize_col = 0;
  vector<int>* pix_csize_row = 0;
  vector<uint64_t>* pix_usedMask = 0;
  //these were renamed in CMSSW_9_1_0: auto-detect
  bool has910_det_lay = t->GetBranch("pix_det") == nullptr;
  if (has910_det_lay) {
    t->SetBranchAddress("pix_subdet", &pix_det);
    t->SetBranchAddress("pix_layer", &pix_lay);
  } else {
    t->SetBranchAddress("pix_det", &pix_det);
    t->SetBranchAddress("pix_lay", &pix_lay);
  }
  t->SetBranchAddress("pix_detId", &pix_detId);
  t->SetBranchAddress("pix_x", &pix_x);
  t->SetBranchAddress("pix_y", &pix_y);
  t->SetBranchAddress("pix_z", &pix_z);
  t->SetBranchAddress("pix_xx", &pix_xx);
  t->SetBranchAddress("pix_xy", &pix_xy);
  t->SetBranchAddress("pix_yy", &pix_yy);
  t->SetBranchAddress("pix_yz", &pix_yz);
  t->SetBranchAddress("pix_zz", &pix_zz);
  t->SetBranchAddress("pix_zx", &pix_zx);
  t->SetBranchAddress("pix_clustSizeCol", &pix_csize_col);
  t->SetBranchAddress("pix_clustSizeRow", &pix_csize_row);
  if (writeHitIterMasks) {
    t->SetBranchAddress("pix_usedMask", &pix_usedMask);
  }

  vector<vector<int>>* pix_simHitIdx = 0;
  t->SetBranchAddress("pix_simHitIdx", &pix_simHitIdx);
  vector<vector<float>>* pix_chargeFraction = 0;
  t->SetBranchAddress("pix_chargeFraction", &pix_chargeFraction);

  //strip hits
  vector<short>* glu_isBarrel = 0;
  vector<unsigned int>* glu_det = 0;
  vector<unsigned int>* glu_lay = 0;
  vector<unsigned int>* glu_detId = 0;
  vector<int>* glu_monoIdx = 0;
  vector<int>* glu_stereoIdx = 0;
  vector<float>* glu_x = 0;
  vector<float>* glu_y = 0;
  vector<float>* glu_z = 0;
  vector<float>* glu_xx = 0;
  vector<float>* glu_xy = 0;
  vector<float>* glu_yy = 0;
  vector<float>* glu_yz = 0;
  vector<float>* glu_zz = 0;
  vector<float>* glu_zx = 0;
  if (!hasPh2hits) {
    t->SetBranchAddress("glu_isBarrel", &glu_isBarrel);
    if (has910_det_lay) {
      t->SetBranchAddress("glu_subdet", &glu_det);
      t->SetBranchAddress("glu_layer", &glu_lay);
    } else {
      t->SetBranchAddress("glu_det", &glu_det);
      t->SetBranchAddress("glu_lay", &glu_lay);
    }
    t->SetBranchAddress("glu_detId", &glu_detId);
    t->SetBranchAddress("glu_monoIdx", &glu_monoIdx);
    t->SetBranchAddress("glu_stereoIdx", &glu_stereoIdx);
    t->SetBranchAddress("glu_x", &glu_x);
    t->SetBranchAddress("glu_y", &glu_y);
    t->SetBranchAddress("glu_z", &glu_z);
    t->SetBranchAddress("glu_xx", &glu_xx);
    t->SetBranchAddress("glu_xy", &glu_xy);
    t->SetBranchAddress("glu_yy", &glu_yy);
    t->SetBranchAddress("glu_yz", &glu_yz);
    t->SetBranchAddress("glu_zz", &glu_zz);
    t->SetBranchAddress("glu_zx", &glu_zx);
  }

  vector<short>* str_isBarrel = 0;
  vector<short>* str_isStereo = 0;
  vector<unsigned int>* str_det = 0;
  vector<unsigned int>* str_lay = 0;
  vector<unsigned int>* str_detId = 0;
  vector<unsigned int>* str_simType = 0;
  vector<float>* str_x = 0;
  vector<float>* str_y = 0;
  vector<float>* str_z = 0;
  vector<float>* str_xx = 0;
  vector<float>* str_xy = 0;
  vector<float>* str_yy = 0;
  vector<float>* str_yz = 0;
  vector<float>* str_zz = 0;
  vector<float>* str_zx = 0;
  vector<float>* str_chargePerCM = 0;
  vector<int>* str_csize = 0;
  vector<uint64_t>* str_usedMask = 0;
  vector<vector<int>>* str_simHitIdx = 0;
  vector<vector<float>>* str_chargeFraction = 0;
  if (!hasPh2hits) {
    t->SetBranchAddress("str_isBarrel", &str_isBarrel);
    t->SetBranchAddress("str_isStereo", &str_isStereo);
    if (has910_det_lay) {
      t->SetBranchAddress("str_subdet", &str_det);
      t->SetBranchAddress("str_layer", &str_lay);
    } else {
      t->SetBranchAddress("str_det", &str_det);
      t->SetBranchAddress("str_lay", &str_lay);
    }
    t->SetBranchAddress("str_detId", &str_detId);
    t->SetBranchAddress("str_simType", &str_simType);
    t->SetBranchAddress("str_x", &str_x);
    t->SetBranchAddress("str_y", &str_y);
    t->SetBranchAddress("str_z", &str_z);
    t->SetBranchAddress("str_xx", &str_xx);
    t->SetBranchAddress("str_xy", &str_xy);
    t->SetBranchAddress("str_yy", &str_yy);
    t->SetBranchAddress("str_yz", &str_yz);
    t->SetBranchAddress("str_zz", &str_zz);
    t->SetBranchAddress("str_zx", &str_zx);
    t->SetBranchAddress("str_chargePerCM", &str_chargePerCM);
    t->SetBranchAddress("str_clustSize", &str_csize);
    if (writeHitIterMasks) {
      t->SetBranchAddress("str_usedMask", &str_usedMask);
    }

    t->SetBranchAddress("str_simHitIdx", &str_simHitIdx);
    t->SetBranchAddress("str_chargeFraction", &str_chargeFraction);
  }

  vector<unsigned short>* ph2_isLower = 0;
  vector<unsigned short>* ph2_subdet = 0;
  vector<unsigned short>* ph2_layer = 0;
  vector<unsigned int>* ph2_detId = 0;
  vector<unsigned short>* ph2_simType = 0;
  vector<float>* ph2_x = 0;
  vector<float>* ph2_y = 0;
  vector<float>* ph2_z = 0;
  vector<float>* ph2_xx = 0;
  vector<float>* ph2_xy = 0;
  vector<float>* ph2_yy = 0;
  vector<float>* ph2_yz = 0;
  vector<float>* ph2_zz = 0;
  vector<float>* ph2_zx = 0;
  vector<uint64_t>* ph2_usedMask = 0;
  vector<vector<int>>* ph2_simHitIdx = 0;
  if (hasPh2hits && applyCCC)
    std::cout << "WARNING: applyCCC is set for Phase2 inputs: applyCCC will be ignored" << std::endl;
  if (hasPh2hits) {
    t->SetBranchAddress("ph2_isLower", &ph2_isLower);
    t->SetBranchAddress("ph2_subdet", &ph2_subdet);
    t->SetBranchAddress("ph2_layer", &ph2_layer);
    t->SetBranchAddress("ph2_detId", &ph2_detId);
    t->SetBranchAddress("ph2_simType", &ph2_simType);
    t->SetBranchAddress("ph2_x", &ph2_x);
    t->SetBranchAddress("ph2_y", &ph2_y);
    t->SetBranchAddress("ph2_z", &ph2_z);
    t->SetBranchAddress("ph2_xx", &ph2_xx);
    t->SetBranchAddress("ph2_xy", &ph2_xy);
    t->SetBranchAddress("ph2_yy", &ph2_yy);
    t->SetBranchAddress("ph2_yz", &ph2_yz);
    t->SetBranchAddress("ph2_zz", &ph2_zz);
    t->SetBranchAddress("ph2_zx", &ph2_zx);
    if (writeHitIterMasks) {
      t->SetBranchAddress("ph2_usedMask", &ph2_usedMask);
    }
    t->SetBranchAddress("ph2_simHitIdx", &ph2_simHitIdx);
  }
  vector<float> ph2_chargeFraction_dummy(16, 0.f);

  // beam spot
  float bsp_x;
  float bsp_y;
  float bsp_z;
  float bsp_sigmax;
  float bsp_sigmay;
  float bsp_sigmaz;
  t->SetBranchAddress("bsp_x", &bsp_x);
  t->SetBranchAddress("bsp_y", &bsp_y);
  t->SetBranchAddress("bsp_z", &bsp_z);
  t->SetBranchAddress("bsp_sigmax", &bsp_sigmax);
  t->SetBranchAddress("bsp_sigmay", &bsp_sigmay);
  t->SetBranchAddress("bsp_sigmaz", &bsp_sigmaz);

  long long totentries = t->GetEntries();
  long long savedEvents = 0;

  DataFile data_file;
  int outOptions = DataFile::ES_Seeds;
  if (writeRecTracks)
    outOptions |= DataFile::ES_CmsswTracks;
  if (writeHitIterMasks)
    outOptions |= DataFile::ES_HitIterMasks;
  outOptions |= DataFile::ES_BeamSpot;

  if (maxevt < 0)
    maxevt = totentries;
  data_file.openWrite(outputFileName, static_cast<int>(nTotalLayers), std::min(maxevt, totentries), outOptions);

  Event EE(0, static_cast<int>(nTotalLayers));

  int numFailCCC = 0;
  int numTotalStr = 0;
  // gDebug = 8;

  for (long long i = 0; savedEvents < maxevt && i < totentries && i < maxevt; ++i) {
    EE.reset(i);

    cout << "process entry i=" << i << " out of " << totentries << ", saved so far " << savedEvents
         << ", with max=" << maxevt << endl;

    t->GetEntry(i);

    cout << "edm event=" << event << endl;

    auto& bs = EE.beamSpot_;
    bs.x = bsp_x;
    bs.y = bsp_y;
    bs.z = bsp_z;
    bs.sigmaZ = bsp_sigmaz;
    bs.beamWidthX = bsp_sigmax;
    bs.beamWidthY = bsp_sigmay;
    //dxdz and dydz are not in the trackingNtuple at the moment

    if (!hasPh2hits) {
      for (unsigned int istr = 0; istr < str_lay->size(); ++istr) {
        if (str_chargePerCM->at(istr) < cutValueCCC)
          numFailCCC++;
        numTotalStr++;
      }
    }

    auto nSims = sim_q->size();
    if (nSims == 0) {
      cout << "branches not loaded" << endl;
      exit(1);
    }
    if (verbosity > 0)
      std::cout << __FILE__ << " " << __LINE__ << " nSims " << nSims << " nSeeds " << see_q->size() << " nRecT "
                << trk_q->size() << std::endl;

    //find best matching tkIdx from a list of simhits indices
    auto bestTkIdx = [&](std::vector<int> const& shs, std::vector<float> const& shfs, int rhIdx, HitType rhType) {
      //assume that all simhits are associated
      int ibest = -1;
      int shbest = -1;
      float hpbest = -1;
      float tpbest = -1;
      float hfbest = -1;

      float maxfrac = -1;
      int ish = -1;
      int nshs = shs.size();
      for (auto const sh : shs) {
        ish++;
        auto tkidx = simhit_simTrkIdx->at(sh);
        //use only sh with available TP
        if (tkidx < 0)
          continue;

        auto hpx = simhit_px->at(sh);
        auto hpy = simhit_py->at(sh);
        auto hpz = simhit_pz->at(sh);
        auto hp = sqrt(hpx * hpx + hpy * hpy + hpz * hpz);

        //look only at hits with p> 50 MeV
        if (hp < 0.05f)
          continue;

        auto tpx = sim_px->at(tkidx);
        auto tpy = sim_py->at(tkidx);
        auto tpz = sim_pz->at(tkidx);
        auto tp = sqrt(tpx * tpx + tpy * tpy + tpz * tpz);

        //take only hits with hp> 0.5*tp
        if (hp < 0.5 * tp)
          continue;

        //pick tkidx corresponding to max hp/tp; .. this is probably redundant
        if (maxfrac < hp / tp) {
          maxfrac = hp / tp;
          ibest = tkidx;
          shbest = sh;
          hpbest = hp;
          tpbest = tp;
          hfbest = shfs[ish];
        }
      }

      //arbitration: a rechit with one matching sim is matched to sim if it's the first
      //FIXME: SOME BETTER SELECTION CAN BE DONE (it will require some more correlated knowledge)
      if (nshs == 1 && ibest >= 0) {
        auto const& srhIdxV = simhit_hitIdx->at(shbest);
        auto const& srhTypeV = simhit_hitType->at(shbest);
        int ih = -1;
        for (auto itype : srhTypeV) {
          ih++;
          if (HitType(itype) == rhType && srhIdxV[ih] != rhIdx) {
            ibest = -1;
            break;
          }
        }
      }

      if (ibest >= 0 && false) {
        std::cout << " best tkIdx " << ibest << " rh " << rhIdx << " for sh " << shbest << " out of " << shs.size()
                  << " hp " << hpbest << " chF " << hfbest << " tp " << tpbest << " process "
                  << simhit_process->at(shbest) << " particle " << simhit_particle->at(shbest) << std::endl;
        if (rhType == HitType::Strip) {
          std::cout << "    sh " << simhit_x->at(shbest) << ", " << simhit_y->at(shbest) << ", " << simhit_z->at(shbest)
                    << "  rh " << str_x->at(rhIdx) << ", " << str_y->at(rhIdx) << ", " << str_z->at(rhIdx) << std::endl;
        }
      }
      return ibest;
    };

    vector<Track>& simTracks_ = EE.simTracks_;
    vector<int> simTrackIdx_(sim_q->size(), -1);  //keep track of original index in ntuple
    vector<int> seedSimIdx(see_q->size(), -1);
    for (unsigned int isim = 0; isim < sim_q->size(); ++isim) {
      //load sim production vertex data
      auto iVtx = sim_parentVtxIdx->at(isim);
      constexpr float largeValF = 9999.f;
      float sim_prodx = iVtx >= 0 ? simvtx_x->at(iVtx) : largeValF;
      float sim_prody = iVtx >= 0 ? simvtx_y->at(iVtx) : largeValF;
      float sim_prodz = iVtx >= 0 ? simvtx_z->at(iVtx) : largeValF;
      //if (fabs(sim_eta->at(isim))>0.8) continue;

      vector<int> const& trkIdxV = sim_trkIdx->at(isim);

      //if (trkIdx<0) continue;
      //FIXME: CHECK IF THE LOOP AND BEST SELECTION IS NEEDED.
      //Pick the first
      const int trkIdx = trkIdxV.empty() ? -1 : trkIdxV[0];

      int nlay = 0;
      if (trkIdx >= 0) {
        std::vector<int> hitlay(nTotalLayers, 0);
        auto const& hits = trk_hitIdx->at(trkIdx);
        auto const& hitTypes = trk_hitType->at(trkIdx);
        auto nHits = hits.size();
        for (auto ihit = 0U; ihit < nHits; ++ihit) {
          auto ihIdx = hits[ihit];
          auto const ihType = HitType(hitTypes[ihit]);

          switch (ihType) {
            case HitType::Pixel: {
              int ipix = ihIdx;
              if (ipix < 0)
                continue;
              int cmsswlay =
                  lnc.convertLayerNumber(pix_det->at(ipix), pix_lay->at(ipix), useMatched, -1, pix_z->at(ipix) > 0);
              if (cmsswlay >= 0 && cmsswlay < static_cast<int>(nTotalLayers))
                hitlay[cmsswlay]++;
              break;
            }
            case HitType::Strip: {
              int istr = ihIdx;
              if (istr < 0)
                continue;
              int cmsswlay = lnc.convertLayerNumber(
                  str_det->at(istr), str_lay->at(istr), useMatched, str_isStereo->at(istr), str_z->at(istr) > 0);
              if (cmsswlay >= 0 && cmsswlay < static_cast<int>(nTotalLayers))
                hitlay[cmsswlay]++;
              break;
            }
            case HitType::Glued: {
              if (useMatched) {
                int iglu = ihIdx;
                if (iglu < 0)
                  continue;
                int cmsswlay =
                    lnc.convertLayerNumber(glu_det->at(iglu), glu_lay->at(iglu), useMatched, -1, glu_z->at(iglu) > 0);
                if (cmsswlay >= 0 && cmsswlay < static_cast<int>(nTotalLayers))
                  hitlay[cmsswlay]++;
              }
              break;
            }
            case HitType::Phase2OT: {
              int istr = ihIdx;
              if (istr < 0)
                continue;
              int cmsswlay = lnc.convertLayerNumber(
                  ph2_subdet->at(istr), ph2_layer->at(istr), useMatched, ph2_isLower->at(istr), ph2_z->at(istr) > 0);
              if (cmsswlay >= 0 && cmsswlay < static_cast<int>(nTotalLayers))
                hitlay[cmsswlay]++;
              break;
            }
            case HitType::Invalid:
              break;  //FIXME. Skip, really?
            default:
              throw std::logic_error("Track type can not be handled");
          }  //hit type
        }    //hits on track
        for (unsigned int i = 0; i < nTotalLayers; i++)
          if (hitlay[i] > 0)
            nlay++;
      }  //count nlay layers on matching reco track

      //cout << Form("track q=%2i p=(%6.3f, %6.3f, %6.3f) x=(%6.3f, %6.3f, %6.3f) nlay=%i",sim_q->at(isim),sim_px->at(isim),sim_py->at(isim),sim_pz->at(isim),sim_prodx,sim_prody,sim_prodz,nlay) << endl;

      SVector3 pos(sim_prodx, sim_prody, sim_prodz);
      SVector3 mom(sim_px->at(isim), sim_py->at(isim), sim_pz->at(isim));
      SMatrixSym66 err;
      err.At(0, 0) = sim_prodx * sim_prodx;
      err.At(1, 1) = sim_prody * sim_prody;
      err.At(2, 2) = sim_prodz * sim_prodz;
      err.At(3, 3) = sim_px->at(isim) * sim_px->at(isim);
      err.At(4, 4) = sim_py->at(isim) * sim_py->at(isim);
      err.At(5, 5) = sim_pz->at(isim) * sim_pz->at(isim);
      TrackState state(sim_q->at(isim), pos, mom, err);
      state.convertFromCartesianToCCS();
      //create track: store number of reco hits in place of track chi2; fill hits later
      //              set label to be its own index in the output file
      Track track(state, float(nlay), simTracks_.size(), 0, nullptr);
      if (sim_bunchCrossing->at(isim) == 0) {  //in time
        if (sim_event->at(isim) == 0)
          track.setProdType(Track::ProdType::Signal);
        else
          track.setProdType(Track::ProdType::InTimePU);
      } else {
        track.setProdType(Track::ProdType::OutOfTimePU);
      }
      if (trkIdx >= 0) {
        int seedIdx = trk_seedIdx->at(trkIdx);
        // Unused: auto const& shTypes = see_hitType->at(seedIdx);
        seedSimIdx[seedIdx] = simTracks_.size();
      }
      if (cleanSimTracks) {
        if (sim_nValid->at(isim) < cleanSimTrack_minSimHits)
          continue;
        if (cleanSimTrack_minRecHits > 0) {
          int nRecToSimHit = 0;
          for (unsigned int ipix = 0; ipix < pix_lay->size() && nRecToSimHit < cleanSimTrack_minRecHits; ++ipix) {
            int ilay = -1;
            ilay = lnc.convertLayerNumber(pix_det->at(ipix), pix_lay->at(ipix), useMatched, -1, pix_z->at(ipix) > 0);
            if (ilay < 0)
              continue;
            int simTkIdxNt = bestTkIdx(pix_simHitIdx->at(ipix), pix_chargeFraction->at(ipix), ipix, HitType::Pixel);
            if (simTkIdxNt >= 0)
              nRecToSimHit++;
          }
          if (hasPh2hits) {
            for (unsigned int istr = 0; istr < ph2_layer->size() && nRecToSimHit < cleanSimTrack_minRecHits; ++istr) {
              int ilay = -1;
              ilay = lnc.convertLayerNumber(
                  ph2_subdet->at(istr), ph2_layer->at(istr), useMatched, ph2_isLower->at(istr), ph2_z->at(istr) > 0);
              if (useMatched && !ph2_isLower->at(istr))
                continue;
              if (ilay == -1)
                continue;
              int simTkIdxNt = bestTkIdx(ph2_simHitIdx->at(istr), ph2_chargeFraction_dummy, istr, HitType::Phase2OT);
              if (simTkIdxNt >= 0)
                nRecToSimHit++;
            }
          } else {
            if (useMatched) {
              for (unsigned int iglu = 0; iglu < glu_lay->size() && nRecToSimHit < cleanSimTrack_minRecHits; ++iglu) {
                if (glu_isBarrel->at(iglu) == 0)
                  continue;
                int igluMono = glu_monoIdx->at(iglu);
                int simTkIdxNt =
                    bestTkIdx(str_simHitIdx->at(igluMono), str_chargeFraction->at(igluMono), igluMono, HitType::Strip);
                if (simTkIdxNt >= 0)
                  nRecToSimHit++;
              }
            }
            for (unsigned int istr = 0; istr < str_lay->size() && nRecToSimHit < cleanSimTrack_minRecHits; ++istr) {
              int ilay = -1;
              ilay = lnc.convertLayerNumber(
                  str_det->at(istr), str_lay->at(istr), useMatched, str_isStereo->at(istr), str_z->at(istr) > 0);
              if (useMatched && str_isBarrel->at(istr) == 1 && str_isStereo->at(istr))
                continue;
              if (ilay == -1)
                continue;
              int simTkIdxNt = bestTkIdx(str_simHitIdx->at(istr), str_chargeFraction->at(istr), istr, HitType::Strip);
              if (simTkIdxNt >= 0)
                nRecToSimHit++;
            }
          }
          if (nRecToSimHit < cleanSimTrack_minRecHits)
            continue;
        }  //count rec-to-sim hits
      }    //cleanSimTracks

      simTrackIdx_[isim] = simTracks_.size();
      simTracks_.push_back(track);
    }

    if (simTracks_.empty() and not writeAllEvents)
      continue;

    vector<Track>& seedTracks_ = EE.seedTracks_;
    vector<vector<int>> pixHitSeedIdx(pix_lay->size());
    vector<vector<int>> strHitSeedIdx(hasPh2hits ? 0 : str_lay->size());
    vector<vector<int>> gluHitSeedIdx(hasPh2hits ? 0 : glu_lay->size());
    vector<vector<int>> ph2HitSeedIdx(hasPh2hits ? ph2_layer->size() : 0);
    for (unsigned int is = 0; is < see_q->size(); ++is) {
      auto isAlgo = TrackAlgorithm(see_algo->at(is));
      if (not allSeeds)
        if (isAlgo != TrackAlgorithm::initialStep && isAlgo != TrackAlgorithm::hltIter0)
          continue;  //select seed in acceptance
      //if (see_pt->at(is)<0.5 || fabs(see_eta->at(is))>0.8) continue;//select seed in acceptance
      SVector3 pos = SVector3(see_stateTrajGlbX->at(is), see_stateTrajGlbY->at(is), see_stateTrajGlbZ->at(is));
      SVector3 mom = SVector3(see_stateTrajGlbPx->at(is), see_stateTrajGlbPy->at(is), see_stateTrajGlbPz->at(is));
      SMatrixSym66 err;
      if (hasCartCov) {
        err.At(0, 0) = see_stateCcov00->at(is);
        err.At(0, 1) = see_stateCcov01->at(is);
        err.At(0, 2) = see_stateCcov02->at(is);
        err.At(0, 3) = see_stateCcov03->at(is);
        err.At(0, 4) = see_stateCcov04->at(is);
        err.At(0, 5) = see_stateCcov05->at(is);
        err.At(1, 1) = see_stateCcov11->at(is);
        err.At(1, 2) = see_stateCcov12->at(is);
        err.At(1, 3) = see_stateCcov13->at(is);
        err.At(1, 4) = see_stateCcov14->at(is);
        err.At(1, 5) = see_stateCcov15->at(is);
        err.At(2, 2) = see_stateCcov22->at(is);
        err.At(2, 3) = see_stateCcov23->at(is);
        err.At(2, 4) = see_stateCcov24->at(is);
        err.At(2, 5) = see_stateCcov25->at(is);
        err.At(3, 3) = see_stateCcov33->at(is);
        err.At(3, 4) = see_stateCcov34->at(is);
        err.At(3, 5) = see_stateCcov35->at(is);
        err.At(4, 4) = see_stateCcov44->at(is);
        err.At(4, 5) = see_stateCcov45->at(is);
        err.At(5, 5) = see_stateCcov55->at(is);
      } else {
        auto const& vCov = see_stateCurvCov->at(is);
        assert(vCov.size() == 15);
        auto vCovP = vCov.begin();
        for (int i = 0; i < 5; ++i)
          for (int j = 0; j <= i; ++j)
            err.At(i, j) = *(vCovP++);
      }
      TrackState state(see_q->at(is), pos, mom, err);
      if (hasCartCov)
        state.convertFromCartesianToCCS();
      else
        state.convertFromGlbCurvilinearToCCS();
      Track track(state, 0, seedSimIdx[is], 0, nullptr);
      track.setAlgorithm(isAlgo);
      auto const& shTypes = see_hitType->at(is);
      auto const& shIdxs = see_hitIdx->at(is);
      if (not allSeeds)
        if (!((isAlgo == TrackAlgorithm::initialStep || isAlgo == TrackAlgorithm::hltIter0) &&
              std::count(shTypes.begin(), shTypes.end(), int(HitType::Pixel)) >= 3))
          continue;  //check algo and nhits
      for (unsigned int ip = 0; ip < shTypes.size(); ip++) {
        unsigned int hidx = shIdxs[ip];
        switch (HitType(shTypes[ip])) {
          case HitType::Pixel: {
            pixHitSeedIdx[hidx].push_back(seedTracks_.size());
            break;
          }
          case HitType::Strip: {
            strHitSeedIdx[hidx].push_back(seedTracks_.size());
            break;
          }
          case HitType::Glued: {
            if (not useMatched) {
              //decompose
              int uidx = glu_monoIdx->at(hidx);
              strHitSeedIdx[uidx].push_back(seedTracks_.size());
              uidx = glu_stereoIdx->at(hidx);
              strHitSeedIdx[uidx].push_back(seedTracks_.size());
            } else {
              gluHitSeedIdx[hidx].push_back(seedTracks_.size());
            }
            break;
          }
          case HitType::Phase2OT: {
            ph2HitSeedIdx[hidx].push_back(seedTracks_.size());
            break;
          }
          case HitType::Invalid:
            break;  //FIXME. Skip, really?
          default:
            throw std::logic_error("Track hit type can not be handled");
        }  //switch( HitType
      }
      seedTracks_.push_back(track);
    }

    if (seedTracks_.empty() and not writeAllEvents)
      continue;

    vector<Track>& cmsswTracks_ = EE.cmsswTracks_;
    vector<vector<int>> pixHitRecIdx(pix_lay->size());
    vector<vector<int>> strHitRecIdx(hasPh2hits ? 0 : str_lay->size());
    vector<vector<int>> gluHitRecIdx(hasPh2hits ? 0 : glu_lay->size());
    vector<vector<int>> ph2HitRecIdx(hasPh2hits ? ph2_layer->size() : 0);
    for (unsigned int ir = 0; ir < trk_q->size(); ++ir) {
      //check the origin; redundant for initialStep ntuples
      if (not allSeeds)
        if ((trk_algoMask->at(ir) & ((1 << int(TrackAlgorithm::initialStep)) | (1 << int(TrackAlgorithm::hltIter0)))) ==
            0) {
          if (verbosity > 1) {
            std::cout << "track " << ir << " failed algo selection for " << int(TrackAlgorithm::initialStep)
                      << ": mask " << trk_algoMask->at(ir) << " origAlgo " << trk_originalAlgo->at(ir) << " algo "
                      << trk_algo->at(ir) << std::endl;
          }
          continue;
        }
      //fill the state in CCS upfront
      SMatrixSym66 err;
      /*	
	vx = -dxy*sin(phi) - pt*cos(phi)/p*pz/p*dz;
	vy =  dxy*cos(phi) - pt*sin(phi)/p*pz/p*dz;
	vz = dz*pt*pt/p/p;
	//partial: ignores cross-terms
	c(vx,vx) = c(dxy,dxy)*sin(phi)*sin(phi) + c(dz,dz)*pow(pt*cos(phi)/p*pz/p ,2);
	c(vx,vy) = -c(dxy,dxy)*cos(phi)*sin(phi) + c(dz,dz)*cos(phi)*sin(phi)*pow(pt/p*pz/p, 2);
	c(vy,vy) = c(dxy,dxy)*cos(phi)*cos(phi) + c(dz,dz)*pow(pt*sin(phi)/p*pz/p ,2);
	c(vx,vz) = -c(dz,dz)*pt*pt/p/p*pt/p*pz/p*cos(phi);
	c(vy,vz) = -c(dz,dz)*pt*pt/p/p*pt/p*pz/p*sin(phi);
	c(vz,vz) = c(dz,dz)*pow(pt*pt/p/p, 2);
      */
      float pt = trk_pt->at(ir);
      float pz = trk_pz->at(ir);
      float p2 = pt * pt + pz * pz;
      float phi = trk_phi->at(ir);
      float sP = sin(phi);
      float cP = cos(phi);
      float dxyErr2 = trk_dxyErr->at(ir);
      dxyErr2 *= dxyErr2;
      float dzErr2 = trk_dzErr->at(ir);
      dzErr2 *= dzErr2;
      float dzErrF2 = trk_dzErr->at(ir) * (pt * pz / p2);
      dzErr2 *= dzErr2;
      err.At(0, 0) = dxyErr2 * sP * sP + dzErrF2 * cP * cP;
      err.At(0, 1) = -dxyErr2 * cP * sP + dzErrF2 * cP * sP;
      err.At(1, 1) = dxyErr2 * cP * cP + dzErrF2 * sP * sP;
      err.At(0, 2) = -dzErrF2 * cP * pt / pz;
      err.At(1, 2) = -dzErrF2 * sP * pt / pz;
      err.At(2, 2) = dzErr2 * std::pow((pt * pt / p2), 2);
      err.At(3, 3) = std::pow(trk_ptErr->at(ir) / pt / pt, 2);
      err.At(4, 4) = std::pow(trk_phiErr->at(ir), 2);
      err.At(5, 5) = std::pow(trk_lambdaErr->at(ir), 2);
      SVector3 pos = SVector3(trk_refpoint_x->at(ir), trk_refpoint_y->at(ir), trk_refpoint_z->at(ir));
      SVector3 mom = SVector3(1.f / pt, phi, M_PI_2 - trk_lambda->at(ir));
      TrackState state(trk_q->at(ir), pos, mom, err);
      Track track(state, trk_nChi2->at(ir), trk_seedIdx->at(ir), 0, nullptr);  //hits are filled later
      track.setAlgorithm(TrackAlgorithm(trk_originalAlgo->at(ir)));
      auto const& hTypes = trk_hitType->at(ir);
      auto const& hIdxs = trk_hitIdx->at(ir);
      for (unsigned int ip = 0; ip < hTypes.size(); ip++) {
        unsigned int hidx = hIdxs[ip];
        switch (HitType(hTypes[ip])) {
          case HitType::Pixel: {
            //cout << "pix=" << hidx << " track=" << cmsswTracks_.size() << endl;
            pixHitRecIdx[hidx].push_back(cmsswTracks_.size());
            break;
          }
          case HitType::Strip: {
            //cout << "str=" << hidx << " track=" << cmsswTracks_.size() << endl;
            strHitRecIdx[hidx].push_back(cmsswTracks_.size());
            break;
          }
          case HitType::Glued: {
            if (not useMatched)
              throw std::logic_error("Tracks have glued hits, but matchedHit load is not configured");
            //cout << "glu=" << hidx << " track=" << cmsswTracks_.size() << endl;
            gluHitRecIdx[hidx].push_back(cmsswTracks_.size());
            break;
          }
          case HitType::Phase2OT: {
            //cout << "ph2=" << hidx << " track=" << cmsswTracks_.size() << endl;
            ph2HitRecIdx[hidx].push_back(cmsswTracks_.size());
            break;
          }
          case HitType::Invalid:
            break;  //FIXME. Skip, really?
          default:
            throw std::logic_error("Track hit type can not be handled");
        }  //switch( HitType
      }
      cmsswTracks_.push_back(track);
    }

    vector<vector<Hit>>& layerHits_ = EE.layerHits_;
    vector<vector<uint64_t>>& layerHitMasks_ = EE.layerHitMasks_;
    vector<MCHitInfo>& simHitsInfo_ = EE.simHitsInfo_;
    int totHits = 0;
    layerHits_.resize(nTotalLayers);
    layerHitMasks_.resize(nTotalLayers);
    for (unsigned int ipix = 0; ipix < pix_lay->size(); ++ipix) {
      int ilay = -1;
      ilay = lnc.convertLayerNumber(pix_det->at(ipix), pix_lay->at(ipix), useMatched, -1, pix_z->at(ipix) > 0);
      if (ilay < 0)
        continue;

      unsigned int imoduleid = tkinfo[ilay].short_id(pix_detId->at(ipix));

      int simTkIdxNt = bestTkIdx(pix_simHitIdx->at(ipix), pix_chargeFraction->at(ipix), ipix, HitType::Pixel);
      int simTkIdx = simTkIdxNt >= 0 ? simTrackIdx_[simTkIdxNt] : -1;  //switch to index in simTracks_
      //cout << Form("pix lay=%i det=%i x=(%6.3f, %6.3f, %6.3f)",ilay+1,pix_det->at(ipix),pix_x->at(ipix),pix_y->at(ipix),pix_z->at(ipix)) << endl;
      SVector3 pos(pix_x->at(ipix), pix_y->at(ipix), pix_z->at(ipix));
      SMatrixSym33 err;
      err.At(0, 0) = pix_xx->at(ipix);
      err.At(1, 1) = pix_yy->at(ipix);
      err.At(2, 2) = pix_zz->at(ipix);
      err.At(0, 1) = pix_xy->at(ipix);
      err.At(0, 2) = pix_zx->at(ipix);
      err.At(1, 2) = pix_yz->at(ipix);
      if (simTkIdx >= 0) {
        simTracks_[simTkIdx].addHitIdx(layerHits_[ilay].size(), ilay, 0);
      }
      for (unsigned int is = 0; is < pixHitSeedIdx[ipix].size(); is++) {
        //cout << "xxx ipix=" << ipix << " seed=" << pixHitSeedIdx[ipix][is] << endl;
        seedTracks_[pixHitSeedIdx[ipix][is]].addHitIdx(layerHits_[ilay].size(), ilay, 0);  //per-hit chi2 is not known
      }
      for (unsigned int ir = 0; ir < pixHitRecIdx[ipix].size(); ir++) {
        //cout << "xxx ipix=" << ipix << " recTrack=" << pixHitRecIdx[ipix][ir] << endl;
        cmsswTracks_[pixHitRecIdx[ipix][ir]].addHitIdx(layerHits_[ilay].size(), ilay, 0);  //per-hit chi2 is not known
      }
      Hit hit(pos, err, totHits);
      hit.setupAsPixel(imoduleid, pix_csize_row->at(ipix), pix_csize_col->at(ipix));
      layerHits_[ilay].push_back(hit);
      if (writeHitIterMasks)
        layerHitMasks_[ilay].push_back(pix_usedMask->at(ipix));
      MCHitInfo hitInfo(simTkIdx, ilay, layerHits_[ilay].size() - 1, totHits);
      simHitsInfo_.push_back(hitInfo);
      totHits++;
    }

    if (hasPh2hits) {
      vector<int> ph2Idx(ph2_layer->size());
      for (unsigned int iph2 = 0; iph2 < ph2_layer->size(); ++iph2) {
        int ilay = -1;
        ilay = lnc.convertLayerNumber(
            ph2_subdet->at(iph2), ph2_layer->at(iph2), useMatched, ph2_isLower->at(iph2), ph2_z->at(iph2) > 0);
        if (useMatched && !ph2_isLower->at(iph2))
          continue;
        if (ilay == -1)
          continue;

        unsigned int imoduleid = tkinfo[ilay].short_id(ph2_detId->at(iph2));

        int simTkIdxNt = bestTkIdx(ph2_simHitIdx->at(iph2), ph2_chargeFraction_dummy, iph2, HitType::Phase2OT);
        int simTkIdx = simTkIdxNt >= 0 ? simTrackIdx_[simTkIdxNt] : -1;  //switch to index in simTracks_

        SVector3 pos(ph2_x->at(iph2), ph2_y->at(iph2), ph2_z->at(iph2));
        SMatrixSym33 err;
        err.At(0, 0) = ph2_xx->at(iph2);
        err.At(1, 1) = ph2_yy->at(iph2);
        err.At(2, 2) = ph2_zz->at(iph2);
        err.At(0, 1) = ph2_xy->at(iph2);
        err.At(0, 2) = ph2_zx->at(iph2);
        err.At(1, 2) = ph2_yz->at(iph2);
        if (simTkIdx >= 0) {
          simTracks_[simTkIdx].addHitIdx(layerHits_[ilay].size(), ilay, 0);
        }
        for (unsigned int ir = 0; ir < ph2HitSeedIdx[iph2].size(); ir++) {
          //cout << "xxx iph2=" << iph2 << " seed=" << ph2HitSeedIdx[iph2][ir] << endl;
          seedTracks_[ph2HitSeedIdx[iph2][ir]].addHitIdx(layerHits_[ilay].size(), ilay, 0);  //per-hit chi2 is not known
        }
        for (unsigned int ir = 0; ir < ph2HitRecIdx[iph2].size(); ir++) {
          //cout << "xxx iph2=" << iph2 << " recTrack=" << ph2HitRecIdx[iph2][ir] << endl;
          cmsswTracks_[ph2HitRecIdx[iph2][ir]].addHitIdx(layerHits_[ilay].size(), ilay, 0);  //per-hit chi2 is not known
        }
        Hit hit(pos, err, totHits);
        hit.setupAsStrip(imoduleid, 0, 1);
        layerHits_[ilay].push_back(hit);
        if (writeHitIterMasks)
          layerHitMasks_[ilay].push_back(ph2_usedMask->at(iph2));
        MCHitInfo hitInfo(simTkIdx, ilay, layerHits_[ilay].size() - 1, totHits);
        simHitsInfo_.push_back(hitInfo);
        totHits++;
      }
    } else {
      if (useMatched) {
        for (unsigned int iglu = 0; iglu < glu_lay->size(); ++iglu) {
          if (glu_isBarrel->at(iglu) == 0)
            continue;
          int igluMono = glu_monoIdx->at(iglu);
          int simTkIdxNt =
              bestTkIdx(str_simHitIdx->at(igluMono), str_chargeFraction->at(igluMono), igluMono, HitType::Strip);
          int simTkIdx = simTkIdxNt >= 0 ? simTrackIdx_[simTkIdxNt] : -1;  //switch to index in simTracks_

          int ilay = lnc.convertLayerNumber(glu_det->at(iglu), glu_lay->at(iglu), useMatched, -1, glu_z->at(iglu) > 0);
          // cout << Form("glu lay=%i det=%i bar=%i x=(%6.3f, %6.3f, %6.3f)",ilay+1,glu_det->at(iglu),glu_isBarrel->at(iglu),glu_x->at(iglu),glu_y->at(iglu),glu_z->at(iglu)) << endl;
          SVector3 pos(glu_x->at(iglu), glu_y->at(iglu), glu_z->at(iglu));
          SMatrixSym33 err;
          err.At(0, 0) = glu_xx->at(iglu);
          err.At(1, 1) = glu_yy->at(iglu);
          err.At(2, 2) = glu_zz->at(iglu);
          err.At(0, 1) = glu_xy->at(iglu);
          err.At(0, 2) = glu_zx->at(iglu);
          err.At(1, 2) = glu_yz->at(iglu);
          if (simTkIdx >= 0) {
            simTracks_[simTkIdx].addHitIdx(layerHits_[ilay].size(), ilay, 0);
          }
          for (unsigned int ir = 0; ir < gluHitSeedIdx[iglu].size(); ir++) {
            //cout << "xxx iglu=" << iglu << " seed=" << gluHitSeedIdx[iglu][ir] << endl;
            seedTracks_[gluHitSeedIdx[iglu][ir]].addHitIdx(
                layerHits_[ilay].size(), ilay, 0);  //per-hit chi2 is not known
          }
          for (unsigned int ir = 0; ir < gluHitRecIdx[iglu].size(); ir++) {
            //cout << "xxx iglu=" << iglu << " recTrack=" << gluHitRecIdx[iglu][ir] << endl;
            cmsswTracks_[gluHitRecIdx[iglu][ir]].addHitIdx(
                layerHits_[ilay].size(), ilay, 0);  //per-hit chi2 is not known
          }

          // QQQQ module-id-in-layer, adc and phi/theta spans are not done for matched hits.
          // Will we ever use / need this?
          assert(false && "Implement module-ids, cluster adc and spans for matched hits!");

          Hit hit(pos, err, totHits);
          layerHits_[ilay].push_back(hit);
          MCHitInfo hitInfo(simTkIdx, ilay, layerHits_[ilay].size() - 1, totHits);
          simHitsInfo_.push_back(hitInfo);
          totHits++;
        }
      }

      vector<int> strIdx;
      strIdx.resize(str_lay->size());
      for (unsigned int istr = 0; istr < str_lay->size(); ++istr) {
        int ilay = -1;
        ilay = lnc.convertLayerNumber(
            str_det->at(istr), str_lay->at(istr), useMatched, str_isStereo->at(istr), str_z->at(istr) > 0);
        if (useMatched && str_isBarrel->at(istr) == 1 && str_isStereo->at(istr))
          continue;
        if (ilay == -1)
          continue;

        unsigned int imoduleid = tkinfo[ilay].short_id(str_detId->at(istr));

        int simTkIdxNt = bestTkIdx(str_simHitIdx->at(istr), str_chargeFraction->at(istr), istr, HitType::Strip);
        int simTkIdx = simTkIdxNt >= 0 ? simTrackIdx_[simTkIdxNt] : -1;  //switch to index in simTracks_

        bool passCCC = applyCCC ? (str_chargePerCM->at(istr) > cutValueCCC) : true;

        //if (str_onTrack->at(istr)==0) continue;//do not consider hits that are not on track!
        SVector3 pos(str_x->at(istr), str_y->at(istr), str_z->at(istr));
        SMatrixSym33 err;
        err.At(0, 0) = str_xx->at(istr);
        err.At(1, 1) = str_yy->at(istr);
        err.At(2, 2) = str_zz->at(istr);
        err.At(0, 1) = str_xy->at(istr);
        err.At(0, 2) = str_zx->at(istr);
        err.At(1, 2) = str_yz->at(istr);
        if (simTkIdx >= 0) {
          if (passCCC)
            simTracks_[simTkIdx].addHitIdx(layerHits_[ilay].size(), ilay, 0);
          else
            simTracks_[simTkIdx].addHitIdx(-9, ilay, 0);
        }
        for (unsigned int ir = 0; ir < strHitSeedIdx[istr].size(); ir++) {
          //cout << "xxx istr=" << istr << " seed=" << strHitSeedIdx[istr][ir] << endl;
          if (passCCC)
            seedTracks_[strHitSeedIdx[istr][ir]].addHitIdx(
                layerHits_[ilay].size(), ilay, 0);  //per-hit chi2 is not known
          else
            seedTracks_[strHitSeedIdx[istr][ir]].addHitIdx(-9, ilay, 0);
        }
        for (unsigned int ir = 0; ir < strHitRecIdx[istr].size(); ir++) {
          //cout << "xxx istr=" << istr << " recTrack=" << strHitRecIdx[istr][ir] << endl;
          if (passCCC)
            cmsswTracks_[strHitRecIdx[istr][ir]].addHitIdx(
                layerHits_[ilay].size(), ilay, 0);  //per-hit chi2 is not known
          else
            cmsswTracks_[strHitRecIdx[istr][ir]].addHitIdx(-9, ilay, 0);
        }
        if (passCCC) {
          Hit hit(pos, err, totHits);
          hit.setupAsStrip(imoduleid, str_chargePerCM->at(istr), str_csize->at(istr));
          layerHits_[ilay].push_back(hit);
          if (writeHitIterMasks)
            layerHitMasks_[ilay].push_back(str_usedMask->at(istr));
          MCHitInfo hitInfo(simTkIdx, ilay, layerHits_[ilay].size() - 1, totHits);
          simHitsInfo_.push_back(hitInfo);
          totHits++;
        }
      }
    }

    // Seed % hit statistics
    nstot += seedTracks_.size();
    for (unsigned int il = 0; il < layerHits_.size(); ++il) {
      int nh = layerHits_[il].size();
      nhitstot[il] += nh;
    }

    if (verbosity > 0) {
      int nt = simTracks_.size();

      int nl = layerHits_.size();

      int nm = simHitsInfo_.size();

      int ns = seedTracks_.size();

      int nr = cmsswTracks_.size();

      printf("number of simTracks %i\n", nt);
      printf("number of layerHits %i\n", nl);
      printf("number of simHitsInfo %i\n", nm);
      printf("number of seedTracks %i\n", ns);
      printf("number of recTracks %i\n", nr);

      if (verbosity > 1) {
        printf("\n");
        for (int il = 0; il < nl; ++il) {
          int nh = layerHits_[il].size();
          for (int ih = 0; ih < nh; ++ih) {
            printf("lay=%i idx=%i mcid=%i x=(%6.3f, %6.3f, %6.3f) r=%6.3f mask=0x%lx\n",
                   il + 1,
                   ih,
                   layerHits_[il][ih].mcHitID(),
                   layerHits_[il][ih].x(),
                   layerHits_[il][ih].y(),
                   layerHits_[il][ih].z(),
                   sqrt(pow(layerHits_[il][ih].x(), 2) + pow(layerHits_[il][ih].y(), 2)),
                   writeHitIterMasks ? layerHitMasks_[il][ih] : 0);
          }
        }

        for (int i = 0; i < nt; ++i) {
          float spt = sqrt(pow(simTracks_[i].px(), 2) + pow(simTracks_[i].py(), 2));
          printf(
              "sim track id=%i q=%2i p=(%6.3f, %6.3f, %6.3f) x=(%6.3f, %6.3f, %6.3f) pT=%7.4f nTotal=%i nFound=%i \n",
              i,
              simTracks_[i].charge(),
              simTracks_[i].px(),
              simTracks_[i].py(),
              simTracks_[i].pz(),
              simTracks_[i].x(),
              simTracks_[i].y(),
              simTracks_[i].z(),
              spt,
              simTracks_[i].nTotalHits(),
              simTracks_[i].nFoundHits());
          int nh = simTracks_[i].nTotalHits();
          for (int ih = 0; ih < nh; ++ih) {
            int hidx = simTracks_[i].getHitIdx(ih);
            int hlay = simTracks_[i].getHitLyr(ih);
            float hx = layerHits_[hlay][hidx].x();
            float hy = layerHits_[hlay][hidx].y();
            float hz = layerHits_[hlay][hidx].z();
            printf("track #%4i hit #%2i idx=%4i lay=%2i x=(% 8.3f, % 8.3f, % 8.3f) r=%8.3f\n",
                   i,
                   ih,
                   hidx,
                   hlay,
                   hx,
                   hy,
                   hz,
                   sqrt(hx * hx + hy * hy));
          }
        }

        for (int i = 0; i < ns; ++i) {
          printf("seed id=%i label=%i algo=%i q=%2i pT=%6.3f p=(%6.3f, %6.3f, %6.3f) x=(%6.3f, %6.3f, %6.3f)\n",
                 i,
                 seedTracks_[i].label(),
                 (int)seedTracks_[i].algorithm(),
                 seedTracks_[i].charge(),
                 seedTracks_[i].pT(),
                 seedTracks_[i].px(),
                 seedTracks_[i].py(),
                 seedTracks_[i].pz(),
                 seedTracks_[i].x(),
                 seedTracks_[i].y(),
                 seedTracks_[i].z());
          int nh = seedTracks_[i].nTotalHits();
          for (int ih = 0; ih < nh; ++ih)
            printf("seed #%i hit #%i idx=%i\n", i, ih, seedTracks_[i].getHitIdx(ih));
        }

        if (writeRecTracks) {
          for (int i = 0; i < nr; ++i) {
            float spt = sqrt(pow(cmsswTracks_[i].px(), 2) + pow(cmsswTracks_[i].py(), 2));
            printf(
                "rec track id=%i label=%i algo=%i chi2=%6.3f q=%2i p=(%6.3f, %6.3f, %6.3f) x=(%6.3f, %6.3f, %6.3f) "
                "pT=%7.4f nTotal=%i nFound=%i \n",
                i,
                cmsswTracks_[i].label(),
                (int)cmsswTracks_[i].algorithm(),
                cmsswTracks_[i].chi2(),
                cmsswTracks_[i].charge(),
                cmsswTracks_[i].px(),
                cmsswTracks_[i].py(),
                cmsswTracks_[i].pz(),
                cmsswTracks_[i].x(),
                cmsswTracks_[i].y(),
                cmsswTracks_[i].z(),
                spt,
                cmsswTracks_[i].nTotalHits(),
                cmsswTracks_[i].nFoundHits());
            int nh = cmsswTracks_[i].nTotalHits();
            for (int ih = 0; ih < nh; ++ih) {
              int hidx = cmsswTracks_[i].getHitIdx(ih);
              int hlay = cmsswTracks_[i].getHitLyr(ih);
              float hx = layerHits_[hlay][hidx].x();
              float hy = layerHits_[hlay][hidx].y();
              float hz = layerHits_[hlay][hidx].z();
              printf("track #%4i hit #%2i idx=%4i lay=%2i x=(% 8.3f, % 8.3f, % 8.3f) r=%8.3f\n",
                     i,
                     ih,
                     hidx,
                     hlay,
                     hx,
                     hy,
                     hz,
                     sqrt(hx * hx + hy * hy));
            }
          }
        }  //if (writeRecTracks){

      }  //verbosity>1
    }    //verbosity>0
    EE.write_out(data_file);

    savedEvents++;
    printf("end of event %lli\n", savedEvents);
  }

  data_file.CloseWrite(savedEvents);
  printf("\nSaved %lli events\n\n", savedEvents);

  printf("Average number of seeds per event %f\n", float(nstot) / float(savedEvents));
  for (unsigned int il = 0; il < nhitstot.size(); ++il)
    printf("Average number of hits in layer %3i = %7.2f\n",
           il,
           float(nhitstot[il]) / float(savedEvents));  //Includes those that failed the cluster charge cut

  if (!hasPh2hits)
    printf("Out of %i hits, %i failed the cut", numTotalStr, numFailCCC);

  //========================================================================

  printf("\n\n================================================================\n");
  printf("=== Max module id for %u layers\n", nTotalLayers);
  printf("================================================================\n");
  for (unsigned int ii = 0; ii < nTotalLayers; ++ii) {
    printf("Layer%2d : %d\n", ii, tkinfo[ii].n_modules());
  }
}
