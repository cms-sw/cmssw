#ifndef L1Trigger_DTTriggerPhase2_MPCleanHitsFilter_h
#define L1Trigger_DTTriggerPhase2_MPCleanHitsFilter_h

#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MPCleanHitsFilter : public MPFilter {
public:
  // Constructors and destructor
  MPCleanHitsFilter(const edm::ParameterSet& pset);
  ~MPCleanHitsFilter() override = default;

  // Main methods
  void initialise(const edm::EventSetup& iEventSetup) override{};
  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           std::vector<cmsdt::metaPrimitive>& inMPath,
           std::vector<cmsdt::metaPrimitive>& outMPath) override{};

  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           MuonPathPtrs& inMPath,
           MuonPathPtrs& outMPath) override;

  void finish() override{};

  // Other public methods
  void removeOutliers(MuonPathPtr& mpath);

  double getMeanTime(MuonPathPtr& mpath);

  void setTimeTolerance(int time) { timeTolerance_ = time; }
  int getTimeTolerance() { return timeTolerance_; }

private:
  // Private attributes
  const bool debug_;
  int timeTolerance_;
};

#endif
