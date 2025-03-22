#ifndef AccessHelper_h
#define AccessHelper_h

#include <vector>
#include <tuple>
#include "LSTEvent.h"

using LSTEvent = ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTEvent;

// ----* Hit *----
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> convertHitsToHitIdxsAndHitTypes(
    LSTEvent* event, std::vector<unsigned int> hits);

// ----* pLS *----
std::vector<unsigned int> getPixelHitsFrompLS(LSTEvent* event, unsigned int pLS);
std::vector<unsigned int> getPixelHitIdxsFrompLS(LSTEvent* event, unsigned int pLS);
std::vector<unsigned int> getPixelHitTypesFrompLS(LSTEvent* event, unsigned int pLS);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFrompLS(LSTEvent* event,
                                                                                              unsigned pLS);

// ----* MD *----
std::vector<unsigned int> getHitsFromMD(LSTEvent* event, unsigned int MD);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromMD(LSTEvent* event,
                                                                                             unsigned MD);

// ----* LS *----
std::vector<unsigned int> getMDsFromLS(LSTEvent* event, unsigned int LS);
std::vector<unsigned int> getHitsFromLS(LSTEvent* event, unsigned int LS);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromLS(LSTEvent* event,
                                                                                             unsigned LS);

// ----* T3 *----
std::vector<unsigned int> getLSsFromT3(LSTEvent* event, unsigned int T3);
std::vector<unsigned int> getMDsFromT3(LSTEvent* event, unsigned int T3);
std::vector<unsigned int> getHitsFromT3(LSTEvent* event, unsigned int T3);
std::vector<unsigned int> getHitTypesFromT3(LSTEvent* event, unsigned int T3);
std::vector<unsigned int> getModuleIdxsFromT3(LSTEvent* event, unsigned int T3);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromT3(LSTEvent* event,
                                                                                             unsigned T3);

// ----* T5 *----
std::vector<unsigned int> getT3sFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getLSsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getMDsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getHitsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getHitIdxsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getHitTypesFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getModuleIdxsFromT5(LSTEvent* event, unsigned int T5);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromT5(LSTEvent* event,
                                                                                             unsigned T5);

// ----* pT3 *----
unsigned int getPixelLSFrompT3(LSTEvent* event, unsigned int pT3);
unsigned int getT3FrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getLSsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getMDsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getOuterTrackerHitsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getPixelHitsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getHitsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getHitIdxsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getHitTypesFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getModuleIdxsFrompT3(LSTEvent* event, unsigned int pT3);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFrompT3(LSTEvent* event,
                                                                                              unsigned pT3);

// ----* pT5 *----
unsigned int getPixelLSFrompT5(LSTEvent* event, unsigned int pT5);
unsigned int getT5FrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getT3sFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getLSsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getMDsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getOuterTrackerHitsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getPixelHitsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getHitsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getHitIdxsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getHitTypesFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getModuleIdxsFrompT5(LSTEvent* event, unsigned int pT5);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFrompT5(LSTEvent* event,
                                                                                              unsigned pT5);

// ----* TC *----
std::vector<unsigned int> getLSsFromTC(LSTEvent* event, unsigned int TC);
std::vector<unsigned int> getHitsFromTC(LSTEvent* event, unsigned int TC);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromTC(LSTEvent* event,
                                                                                             unsigned int TC);

#endif
