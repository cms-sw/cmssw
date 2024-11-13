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
using ::lst::PixelType;

// --------------------- ======================== ---------------------

bool goodEvent();
float runMiniDoublet(LSTEvent *event, int evt);
float runSegment(LSTEvent *event);
float runT4(LSTEvent *event);
float runT4x(LSTEvent *event);
float runpT4(LSTEvent *event);
float runT3(LSTEvent *event);
float runTrackCandidate(LSTEvent *event, bool no_pls_dupclean, bool tc_pls_triplets);
float runQuintuplet(LSTEvent *event);
float runPixelQuintuplet(LSTEvent *event);
float runPixelLineSegment(LSTEvent *event, bool no_pls_dupclean);
float runpT3(LSTEvent *event);

// --------------------- ======================== ---------------------

std::vector<int> matchedSimTrkIdxs(std::vector<unsigned int> hitidxs,
                                   std::vector<unsigned int> hittypes,
                                   bool verbose = false);
std::vector<int> matchedSimTrkIdxs(std::vector<int> hitidxs, std::vector<int> hittypes, bool verbose = false);
int getDenomSimTrkType(int isimtrk);
int getDenomSimTrkType(std::vector<int> simidxs);

// --------------------- ======================== ---------------------

float drfracSimHitConsistentWithHelix(int isimtrk, int isimhitidx);
float drfracSimHitConsistentWithHelix(lst_math::Helix &helix, int isimhitidx);
float distxySimHitConsistentWithHelix(int isimtrk, int isimhitidx);
float distxySimHitConsistentWithHelix(lst_math::Helix &helix, int isimhitidx);
TVector3 calculateR3FromPCA(const TVector3 &p3, const float dxy, const float dz);

// --------------------- ======================== ---------------------

void addInputsToLineSegmentTrackingPreLoad(std::vector<std::vector<float>> &out_trkX,
                                           std::vector<std::vector<float>> &out_trkY,
                                           std::vector<std::vector<float>> &out_trkZ,
                                           std::vector<std::vector<unsigned int>> &out_hitId,
                                           std::vector<std::vector<unsigned int>> &out_hitIdxs,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec0,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec1,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec2,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec3,
                                           std::vector<std::vector<float>> &out_deltaPhi_vec,
                                           std::vector<std::vector<float>> &out_ptIn_vec,
                                           std::vector<std::vector<float>> &out_ptErr_vec,
                                           std::vector<std::vector<float>> &out_px_vec,
                                           std::vector<std::vector<float>> &out_py_vec,
                                           std::vector<std::vector<float>> &out_pz_vec,
                                           std::vector<std::vector<float>> &out_eta_vec,
                                           std::vector<std::vector<float>> &out_etaErr_vec,
                                           std::vector<std::vector<float>> &out_phi_vec,
                                           std::vector<std::vector<int>> &out_charge_vec,
                                           std::vector<std::vector<unsigned int>> &out_seedIdx_vec,
                                           std::vector<std::vector<int>> &out_superbin_vec,
                                           std::vector<std::vector<PixelType>> &out_pixelType_vec,
                                           std::vector<std::vector<char>> &out_isQuad_vec);

float addInputsToEventPreLoad(LSTEvent *event,
                              bool useOMP,
                              std::vector<float> trkX,
                              std::vector<float> trkY,
                              std::vector<float> trkZ,
                              std::vector<unsigned int> hitId,
                              std::vector<unsigned int> hitIdxs,
                              std::vector<unsigned int> hitIndices_vec0,
                              std::vector<unsigned int> hitIndices_vec1,
                              std::vector<unsigned int> hitIndices_vec2,
                              std::vector<unsigned int> hitIndices_vec3,
                              std::vector<float> deltaPhi_vec,
                              std::vector<float> ptIn_vec,
                              std::vector<float> ptErr_vec,
                              std::vector<float> px_vec,
                              std::vector<float> py_vec,
                              std::vector<float> pz_vec,
                              std::vector<float> eta_vec,
                              std::vector<float> etaErr_vec,
                              std::vector<float> phi_vec,
                              std::vector<int> charge_vec,
                              std::vector<unsigned int> seedIdx_vec,
                              std::vector<int> superbin_vec,
                              std::vector<PixelType> pixelType_vec,
                              std::vector<char> isQuad_vec);

void printTimingInformation(std::vector<std::vector<float>> &timing_information, float fullTime, float fullavg);

// --------------------- ======================== ---------------------

TString get_absolute_path_after_check_file_exists(const std::string name);
void writeMetaData();

// --------------------- ======================== ---------------------

// DEPRECATED FUNCTION
float addInputsToLineSegmentTrackingUsingExplicitMemory(LSTEvent &event);
float addInputsToLineSegmentTracking(LSTEvent &event, bool useOMP);

#endif
