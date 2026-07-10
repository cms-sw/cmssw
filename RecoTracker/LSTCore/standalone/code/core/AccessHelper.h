#ifndef AccessHelper_h
#define AccessHelper_h

#include <vector>
#include <tuple>
#include "LSTEvent.h"

using LSTEvent = ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTEvent;

// ----* Hit *----
std::tuple<std::vector<unsigned int>, std::vector<lst::HitType>> convertHitsToHitIdxsAndHitTypes(
    LSTEvent* event, std::vector<unsigned int> hits);

// ----* pLS *----
std::vector<unsigned int> getHitsFrompLS(LSTEvent* event, unsigned int pLS);
std::vector<unsigned int> getHitIdxsFrompLS(LSTEvent* event, unsigned int pLS);
std::vector<lst::HitType> getHitTypesFrompLS(LSTEvent* event, unsigned int pLS);
std::tuple<std::vector<unsigned int>, std::vector<lst::HitType>> getHitIdxsAndHitTypesFrompLS(LSTEvent* event,
                                                                                              unsigned pLS);

// ----* MD *----
std::vector<unsigned int> getHitsFromMD(LSTEvent* event, unsigned int MD);
std::tuple<std::vector<unsigned int>, std::vector<lst::HitType>> getHitIdxsAndHitTypesFromMD(LSTEvent* event,
                                                                                             unsigned MD);

// ----* LS *----
std::vector<unsigned int> getMDsFromLS(LSTEvent* event, unsigned int LS);
std::vector<unsigned int> getHitsFromLS(LSTEvent* event, unsigned int LS);
std::tuple<std::vector<unsigned int>, std::vector<lst::HitType>> getHitIdxsAndHitTypesFromLS(LSTEvent* event,
                                                                                             unsigned LS);

// ----* T3 *----
std::vector<unsigned int> getLSsFromT3(LSTEvent* event, unsigned int T3);
std::vector<unsigned int> getMDsFromT3(LSTEvent* event, unsigned int T3);
std::vector<unsigned int> getHitsFromT3(LSTEvent* event, unsigned int T3);
std::vector<lst::HitType> getHitTypesFromT3(LSTEvent* event, unsigned int T3);
std::vector<unsigned int> getModuleIdxsFromT3(LSTEvent* event, unsigned int T3);
std::tuple<std::vector<unsigned int>, std::vector<lst::HitType>> getHitIdxsAndHitTypesFromT3(LSTEvent* event,
                                                                                             unsigned T3);

// ----* T4 *----
std::vector<unsigned int> getT3sFromT4(LSTEvent* event, unsigned int T4);
std::vector<unsigned int> getLSsFromT4(LSTEvent* event, unsigned int T4);
std::vector<unsigned int> getMDsFromT4(LSTEvent* event, unsigned int T4);
std::vector<unsigned int> getHitsFromT4(LSTEvent* event, unsigned int T4);
std::vector<unsigned int> getHitIdxsFromT4(LSTEvent* event, unsigned int T4);
std::vector<lst::HitType> getHitTypesFromT4(LSTEvent* event, unsigned int T4);
std::vector<unsigned int> getModuleIdxsFromT4(LSTEvent* event, unsigned int T4);
std::tuple<std::vector<unsigned int>, std::vector<lst::HitType>> getHitIdxsAndHitTypesFromT4(LSTEvent* event,
                                                                                             unsigned T4);

// ----* T5 *----
std::vector<unsigned int> getT3sFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getLSsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getMDsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getHitsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getHitIdxsFromT5(LSTEvent* event, unsigned int T5);
std::vector<lst::HitType> getHitTypesFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getModuleIdxsFromT5(LSTEvent* event, unsigned int T5);
std::tuple<std::vector<unsigned int>, std::vector<lst::HitType>> getHitIdxsAndHitTypesFromT5(LSTEvent* event,
                                                                                             unsigned T5);

// ----* pT3 *----
unsigned int getpLSFrompT3(LSTEvent* event, unsigned int pT3);
unsigned int getT3FrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getLSsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getMDsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getT3HitsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getpLSHitsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getHitsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getHitIdxsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<lst::HitType> getHitTypesFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getModuleIdxsFrompT3(LSTEvent* event, unsigned int pT3);
std::tuple<std::vector<unsigned int>, std::vector<lst::HitType>> getHitIdxsAndHitTypesFrompT3(LSTEvent* event,
                                                                                              unsigned pT3);

// ----* pT5 *----
unsigned int getpLSFrompT5(LSTEvent* event, unsigned int pT5);
unsigned int getT5FrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getT3sFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getLSsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getMDsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getT5HitsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getpLSHitsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getHitsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getHitIdxsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<lst::HitType> getHitTypesFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getModuleIdxsFrompT5(LSTEvent* event, unsigned int pT5);
std::tuple<std::vector<unsigned int>, std::vector<lst::HitType>> getHitIdxsAndHitTypesFrompT5(LSTEvent* event,
                                                                                              unsigned pT5);

// ----* TC *----
std::vector<unsigned int> getLSsFromTC(LSTEvent* event, unsigned int TC);
std::vector<unsigned int> getHitsFromTC(LSTEvent* event, unsigned int TC);
std::tuple<std::vector<unsigned int>, std::vector<lst::HitType>> getUnderlyingHitIdxsAndHitTypesFromTC(LSTEvent* event,
                                                                                                       unsigned int TC);
std::pair<std::vector<unsigned int>, std::vector<lst::HitType>> getHitIdxsAndHitTypesFromTC(LSTEvent* event,
                                                                                            unsigned int tc_idx);

#endif
