#ifndef Phase2L1Trigger_DTTrigger_MPThetaMatching_h
#define Phase2L1Trigger_DTTrigger_MPThetaMatching_h

#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MPThetaMatching : public MPFilter {
public:
  // Constructors and destructor
  MPThetaMatching(const edm::ParameterSet &pset);
  ~MPThetaMatching() override;  // = default;

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &inMPaths,
           std::vector<cmsdt::metaPrimitive> &outMPaths) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &allMPaths,
           std::vector<cmsdt::metaPrimitive> &inMPaths,
           std::vector<cmsdt::metaPrimitive> &outMPaths) override {};
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMPath,
           MuonPathPtrs &outMPath) override {};

  void finish() override;

  // Public attributes

private:
  // Private methods
  std::vector<cmsdt::metaPrimitive> filter(std::vector<cmsdt::metaPrimitive> inMPs, double shift_back);

  bool isThereThetaMPInChamber(int sector, int wheel, int station, std::vector<cmsdt::metaPrimitive> thetaMPs);
  std::vector<cmsdt::metaPrimitive> getBestThetaMPInChamber(std::vector<cmsdt::metaPrimitive> thetaMPs);

  // Function to compare pairs based on the float value, ascending order
  static bool comparePairs(const std::tuple<cmsdt::metaPrimitive, cmsdt::metaPrimitive, float> &a,
                           const std::tuple<cmsdt::metaPrimitive, cmsdt::metaPrimitive, float> &b) {
    return std::get<2>(a) < std::get<2>(b);
  };
  void orderAndSave(std::vector<std::tuple<cmsdt::metaPrimitive, cmsdt::metaPrimitive, float>> deltaTimePosPhiCands,
                    std::vector<cmsdt::metaPrimitive> *outMPaths,
                    std::vector<cmsdt::metaPrimitive> *savedThetas);

  // Private attributes
  const bool debug_;
  int th_option_;
  int th_quality_;
  int scenario_;
};

#endif
