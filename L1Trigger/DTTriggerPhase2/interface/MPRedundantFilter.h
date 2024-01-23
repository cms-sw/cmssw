#ifndef Phase2L1Trigger_DTTrigger_MPRedundantFilter_h
#define Phase2L1Trigger_DTTrigger_MPRedundantFilter_h

#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"

#include <iostream>
#include <fstream>
#include <deque>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MPRedundantFilter : public MPFilter {
public:
  // Constructors and destructor
  MPRedundantFilter(const edm::ParameterSet& pset);
  ~MPRedundantFilter() override;

  // Main methods
  void initialise(const edm::EventSetup& iEventSetup) override;
  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           std::vector<cmsdt::metaPrimitive>& inMPath,
           std::vector<cmsdt::metaPrimitive>& outMPath) override{};
  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           std::vector<cmsdt::metaPrimitive>& inSLMPath,
           std::vector<cmsdt::metaPrimitive>& inCorMPath,
           std::vector<cmsdt::metaPrimitive>& outMPath) override{};
  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           MuonPathPtrs& inMPath,
           MuonPathPtrs& outMPath) override;
  void finish() override { buffer_.clear(); };

  // Other public methods

private:
  void filter(MuonPathPtr& mpath, MuonPathPtrs& outMPaths);
  bool isInBuffer(MuonPathPtr& mpath);

  // Private attributes
  const bool debug_;
  unsigned int maxBufferSize_;
  std::deque<MuonPathPtr> buffer_;
};

#endif
