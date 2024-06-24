#ifndef write_sdl_ntuple_h
#define write_sdl_ntuple_h

#include <iostream>
#include <tuple>

#include "SDLMath.h"
#include "Event.h"

#include "AnalysisConfig.h"
#include "trkCore.h"
#include "AccessHelper.h"

// Common
void createOutputBranches();
void createRequiredOutputBranches();
void createOptionalOutputBranches();
void createGnnNtupleBranches();

void fillOutputBranches(SDL::Event<SDL::Acc>* event);
void setOutputBranches(SDL::Event<SDL::Acc>* event);
void setOptionalOutputBranches(SDL::Event<SDL::Acc>* event);
void setPixelQuintupletOutputBranches(SDL::Event<SDL::Acc>* event);
void setQuintupletOutputBranches(SDL::Event<SDL::Acc>* event);
void setPixelTripletOutputBranches(SDL::Event<SDL::Acc>* event);
void setGnnNtupleBranches(SDL::Event<SDL::Acc>* event);
void setGnnNtupleMiniDoublet(SDL::Event<SDL::Acc>* event, unsigned int MD);

std::tuple<int, float, float, float, int, std::vector<int>> parseTrackCandidate(SDL::Event<SDL::Acc>* event,
                                                                                unsigned int);
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepT5(
    SDL::Event<SDL::Acc>* event, unsigned int);
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepT3(
    SDL::Event<SDL::Acc>* event, unsigned int);
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parseT5(
    SDL::Event<SDL::Acc>* event, unsigned int);
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepLS(
    SDL::Event<SDL::Acc>* event, unsigned int);

// Print multiplicities
void printMiniDoubletMultiplicities(SDL::Event<SDL::Acc>* event);
void printHitMultiplicities(SDL::Event<SDL::Acc>* event);

// Print objects (GPU)
void printAllObjects(SDL::Event<SDL::Acc>* event);
void printpT4s(SDL::Event<SDL::Acc>* event);
void printMDs(SDL::Event<SDL::Acc>* event);
void printLSs(SDL::Event<SDL::Acc>* event);
void printpLSs(SDL::Event<SDL::Acc>* event);
void printT3s(SDL::Event<SDL::Acc>* event);
void printT4s(SDL::Event<SDL::Acc>* event);
void printTCs(SDL::Event<SDL::Acc>* event);

// Print anomalous multiplicities
void debugPrintOutlierMultiplicities(SDL::Event<SDL::Acc>* event);

#endif
