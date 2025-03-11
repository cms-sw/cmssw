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
                                   bool verbose = false,
                                   float *pmatched = nullptr);
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

float addInputsToEventPreLoad(LSTEvent *event,
                              lst::HitsHostCollection *hitsHC,
                              lst::PixelSegmentsHostCollection *pixelSegmentsHC);

void printTimingInformation(std::vector<std::vector<float>> &timing_information, float fullTime, float fullavg);

// --------------------- ======================== ---------------------

TString get_absolute_path_after_check_file_exists(const std::string name);
void writeMetaData();

// --------------------- ======================== ---------------------

// DEPRECATED FUNCTION
float addInputsToLineSegmentTrackingUsingExplicitMemory(LSTEvent &event);
float addInputsToLineSegmentTracking(LSTEvent &event, bool useOMP);

#endif
