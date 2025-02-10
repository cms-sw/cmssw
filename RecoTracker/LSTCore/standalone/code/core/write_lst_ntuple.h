#ifndef write_lst_ntuple_h
#define write_lst_ntuple_h

#include <iostream>
#include <tuple>

#include "lst_math.h"
#include "LSTEvent.h"

#include "AnalysisConfig.h"
#include "trkCore.h"
#include "AccessHelper.h"

using LSTEvent = ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTEvent;

// Common
void createOutputBranches();
void createRequiredOutputBranches();
void createOptionalOutputBranches();
void createGnnNtupleBranches();
void createT5DNNBranches();

void fillOutputBranches(LSTEvent* event);
void setOutputBranches(LSTEvent* event);
void setOptionalOutputBranches(LSTEvent* event);
void setOccupancyBranches(LSTEvent* event);
void setPixelQuintupletOutputBranches(LSTEvent* event);
void setQuintupletOutputBranches(LSTEvent* event);
void setPixelTripletOutputBranches(LSTEvent* event);
void setGnnNtupleBranches(LSTEvent* event);
void setGnnNtupleMiniDoublet(LSTEvent* event, unsigned int MD);
void fillT5DNNBranches(LSTEvent* event, unsigned int T3);
void setT5DNNBranches(LSTEvent* event);

std::tuple<int, float, float, float, int, std::vector<int>> parseTrackCandidate(LSTEvent* event, unsigned int);
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepT5(LSTEvent* event,
                                                                                               unsigned int);
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepT3(LSTEvent* event,
                                                                                               unsigned int);
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parseT5(LSTEvent* event,
                                                                                              unsigned int);
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepLS(LSTEvent* event,
                                                                                               unsigned int);

// Print multiplicities
void printMiniDoubletMultiplicities(LSTEvent* event);
void printHitMultiplicities(LSTEvent* event);

// Print objects (GPU)
void printAllObjects(LSTEvent* event);
void printpT4s(LSTEvent* event);
void printMDs(LSTEvent* event);
void printLSs(LSTEvent* event);
void printpLSs(LSTEvent* event);
void printT3s(LSTEvent* event);
void printT4s(LSTEvent* event);
void printTCs(LSTEvent* event);

#endif
