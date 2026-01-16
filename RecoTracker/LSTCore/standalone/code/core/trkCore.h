#ifndef trkCore_h
#define trkCore_h

#include "LSTEvent.h"

#include "Trktree.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "AnalysisConfig.h"
#include "ModuleConnectionMap.h"
#include "lst_math.h"
#include <numeric>
#include <filesystem>

using LSTEvent = ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTEvent;
using LSTInputDeviceCollection = ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTInputDeviceCollection;
using ::lst::PixelType;

// --------------------- ======================== ---------------------

bool goodEvent();
float runMiniDoublet(LSTEvent* event, int evt);
float runSegment(LSTEvent* event);
float runT4(LSTEvent* event);
float runT4x(LSTEvent* event);
float runT3(LSTEvent* event);
float runTrackCandidate(LSTEvent* event, bool no_pls_dupclean, bool tc_pls_triplets);
float runQuadruplet(LSTEvent* event);
float runQuintuplet(LSTEvent* event);
float runPixelQuintuplet(LSTEvent* event);
float runPixelLineSegment(LSTEvent* event, bool no_pls_dupclean);
float runpT3(LSTEvent* event);

// --------------------- ======================== ---------------------

std::vector<int> matchedSimTrkIdxs(std::vector<unsigned int> hitidxs,
                                   std::vector<unsigned int> hittypes,
                                   std::vector<int> const& trk_simhit_simTrkIdx,
                                   std::vector<std::vector<int>> const& trk_ph2_simHitIdx,
                                   std::vector<std::vector<int>> const& trk_pix_simHitIdx,
                                   bool verbose = false,
                                   float matchfrac = 0.75,
                                   float* pmatched = nullptr);
std::tuple<std::vector<int>, std::vector<float>> matchedSimTrkIdxsAndFracs(
    std::vector<unsigned int> hitidxs,
    std::vector<unsigned int> hittypes,
    std::vector<int> const& trk_simhit_simTrkIdx,
    std::vector<std::vector<int>> const& trk_ph2_simHitIdx,
    std::vector<std::vector<int>> const& trk_pix_simHitIdx,
    bool verbose = false,
    float matchfrac = 0.75,
    float* pmatched = nullptr);
int getDenomSimTrkType(int isimtrk,
                       std::vector<int> const& trk_sim_q,
                       std::vector<float> const& trk_sim_pt,
                       std::vector<float> const& trk_sim_eta,
                       std::vector<int> const& trk_sim_bunchCrossing,
                       std::vector<int> const& trk_sim_event,
                       std::vector<int> const& trk_sim_parentVtxIdx,
                       std::vector<float> const& trk_simvtx_x,
                       std::vector<float> const& trk_simvtx_y,
                       std::vector<float> const& trk_simvtx_z);
int getDenomSimTrkType(std::vector<int> simidxs,
                       std::vector<int> const& trk_sim_q,
                       std::vector<float> const& trk_sim_pt,
                       std::vector<float> const& trk_sim_eta,
                       std::vector<int> const& trk_sim_bunchCrossing,
                       std::vector<int> const& trk_sim_event,
                       std::vector<int> const& trk_sim_parentVtxIdx,
                       std::vector<float> const& trk_simvtx_x,
                       std::vector<float> const& trk_simvtx_y,
                       std::vector<float> const& trk_simvtx_z);

// --------------------- ======================== ---------------------

float drfracSimHitConsistentWithHelix(int isimtrk,
                                      int isimhitidx,
                                      std::vector<float> const& trk_simvtx_x,
                                      std::vector<float> const& trk_simvtx_y,
                                      std::vector<float> const& trk_simvtx_z,
                                      std::vector<float> const& trk_sim_pt,
                                      std::vector<float> const& trk_sim_eta,
                                      std::vector<float> const& trk_sim_phi,
                                      std::vector<int> const& trk_sim_q,
                                      std::vector<float> const& trk_simhit_x,
                                      std::vector<float> const& trk_simhit_y,
                                      std::vector<float> const& trk_simhit_z);
float drfracSimHitConsistentWithHelix(lst_math::Helix& helix,
                                      int isimhitidx,
                                      std::vector<float> const& trk_simhit_x,
                                      std::vector<float> const& trk_simhit_y,
                                      std::vector<float> const& trk_simhit_z);
float distxySimHitConsistentWithHelix(int isimtrk,
                                      int isimhitidx,
                                      std::vector<float> const& trk_simvtx_x,
                                      std::vector<float> const& trk_simvtx_y,
                                      std::vector<float> const& trk_simvtx_z,
                                      std::vector<float> const& trk_sim_pt,
                                      std::vector<float> const& trk_sim_eta,
                                      std::vector<float> const& trk_sim_phi,
                                      std::vector<int> const& trk_sim_q,
                                      std::vector<float> const& trk_simhit_x,
                                      std::vector<float> const& trk_simhit_y,
                                      std::vector<float> const& trk_simhit_z);
float distxySimHitConsistentWithHelix(lst_math::Helix& helix,
                                      int isimhitidx,
                                      std::vector<float> const& trk_simhit_x,
                                      std::vector<float> const& trk_simhit_y,
                                      std::vector<float> const& trk_simhit_z);
TVector3 calculateR3FromPCA(const TVector3& p3, const float dxy, const float dz);

// --------------------- ======================== ---------------------

float addInputsToEventPreLoad(LSTEvent* event,
                              lst::LSTInputHostCollection* lstInputHC,
                              LSTInputDeviceCollection* lstInputDC,
                              ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue);

void printTimingInformation(std::vector<std::vector<float>>& timing_information, float fullTime, float fullavg);

// --------------------- ======================== ---------------------

TString get_absolute_path_after_check_file_exists(const std::string name);
void writeMetaData();

template <typename T>
std::vector<size_t> sort_indices(const std::vector<T>& vec) {
  std::vector<size_t> indices(vec.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&vec](size_t i1, size_t i2) { return vec[i1] > vec[i2]; });
  return indices;
}

// --------------------- ======================== ---------------------

// DEPRECATED FUNCTION
float addInputsToLineSegmentTrackingUsingExplicitMemory(LSTEvent& event);
float addInputsToLineSegmentTracking(LSTEvent& event, bool useOMP);

#endif
