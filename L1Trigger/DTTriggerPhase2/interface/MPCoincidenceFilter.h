#ifndef Phase2L1Trigger_DTTrigger_MPCoincidenceFilter_h
#define Phase2L1Trigger_DTTrigger_MPCoincidenceFilter_h

#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MPCoincidenceFilter : public MPFilter {
public:
  // Constructors and destructor
  MPCoincidenceFilter(const edm::ParameterSet &pset);
  ~MPCoincidenceFilter() override = default;

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &inMPath,
           std::vector<cmsdt::metaPrimitive> &outMPath) override {};
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &allMPaths,
           std::vector<cmsdt::metaPrimitive> &inMPaths,
           std::vector<cmsdt::metaPrimitive> &outMPaths) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMPath,
           MuonPathPtrs &outMPath) override {};

  void finish() override;

  // Other public methods

  // Public attributes

  std::map<std::string, float> mphi_mean{{"wh-2ch1", 1.0},  {"wh-1ch1", 0.9}, {"wh0ch1", -0.3},  {"wh1ch1", 0.9},
                                         {"wh2ch1", 1.0},   {"wh-2ch2", 1.4}, {"wh-1ch2", -0.4}, {"wh0ch2", -0.3},
                                         {"wh1ch2", -0.4},  {"wh2ch2", 1.5},  {"wh-2ch3", -0.1}, {"wh-1ch3", -0.2},
                                         {"wh0ch3", -0.1},  {"wh1ch3", -0.3}, {"wh2ch3", -0.4},  {"wh-2ch4", -1.0},
                                         {"wh-1ch4", -1.1}, {"wh0ch4", -1.0}, {"wh1ch4", -1.1},  {"wh2ch4", -0.8}};

  std::map<std::string, float> mphi_width{{"wh-2ch1", 7.2}, {"wh-1ch1", 7.0}, {"wh0ch1", 11.2}, {"wh1ch1", 7.4},
                                          {"wh2ch1", 7.1},  {"wh-2ch2", 6.6}, {"wh-1ch2", 8.5}, {"wh0ch2", 11.1},
                                          {"wh1ch2", 8.5},  {"wh2ch2", 6.5},  {"wh-2ch3", 8.0}, {"wh-1ch3", 9.2},
                                          {"wh0ch3", 11.2}, {"wh1ch3", 9.1},  {"wh2ch3", 7.8},  {"wh-2ch4", 8.0},
                                          {"wh-1ch4", 9.6}, {"wh0ch4", 11.8}, {"wh1ch4", 9.3},  {"wh2ch4", 7.7}};

  std::map<std::string, float> mth_mean{{"wh-2ch1", -17.4}, {"wh-1ch1", -9.5},  {"wh0ch1", -0.8},   {"wh1ch1", -9.8},
                                        {"wh2ch1", -17.1},  {"wh-2ch2", -18.9}, {"wh-1ch2", -6.6},  {"wh0ch2", 0.5},
                                        {"wh1ch2", -6.8},   {"wh2ch2", -19.2},  {"wh-2ch3", -16.3}, {"wh-1ch3", -3.2},
                                        {"wh0ch3", 1.9},    {"wh1ch3", -3.6},   {"wh2ch3", -17.5},  {"wh-2ch4", 0.0},
                                        {"wh-1ch4", 0.0},   {"wh0ch4", 0.0},    {"wh1ch4", 0.0},    {"wh2ch4", 0.0}};

  std::map<std::string, float> mth_width{{"wh-2ch1", 33.5}, {"wh-1ch1", 12.6}, {"wh0ch1", 10.1},  {"wh1ch1", 14.4},
                                         {"wh2ch1", 44.8},  {"wh-2ch2", 23.0}, {"wh-1ch2", 13.2}, {"wh0ch2", 11.6},
                                         {"wh1ch2", 14.0},  {"wh2ch2", 25.6},  {"wh-2ch3", 22.5}, {"wh-1ch3", 13.8},
                                         {"wh0ch3", 13.9},  {"wh1ch3", 14.2},  {"wh2ch3", 24.2},  {"wh-2ch4", 9.4},
                                         {"wh-1ch4", 9.4},  {"wh0ch4", 9.4},   {"wh1ch4", 9.4},   {"wh2ch4", 9.4}};

private:
  // Private methods
  std::vector<cmsdt::metaPrimitive> filter(std::vector<cmsdt::metaPrimitive> inMPs,
                                           std::vector<cmsdt::metaPrimitive> allMPs,
                                           int co_option,
                                           int co_quality,
                                           int co_wh2option,
                                           double shift_back);

  // Private attributes
  const bool debug_;
  int co_option_;
  int co_quality_;
  int co_wh2option_;
  int scenario_;
};

#endif
