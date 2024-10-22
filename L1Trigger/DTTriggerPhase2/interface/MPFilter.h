#ifndef Phase2L1Trigger_DTTrigger_MPFilter_h
#define Phase2L1Trigger_DTTrigger_MPFilter_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MPFilter {
public:
  // Constructors and destructor
  MPFilter(const edm::ParameterSet& pset);
  virtual ~MPFilter();

  // Main methods
  virtual void initialise(const edm::EventSetup& iEventSetup) = 0;
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   std::vector<cmsdt::metaPrimitive>& inMPath,
                   std::vector<cmsdt::metaPrimitive>& outMPath) = 0;
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   std::vector<cmsdt::metaPrimitive>& inSLMPath,
                   std::vector<cmsdt::metaPrimitive>& inCorMPath,
                   std::vector<cmsdt::metaPrimitive>& outMPath) = 0;
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   MuonPathPtrs& inMPath,
                   MuonPathPtrs& outMPath) = 0;

  virtual void finish() = 0;

  // Other public methods

  // Public attributes
  // max drift velocity
  edm::FileInPath maxdrift_filename_;
  int maxdriftinfo_[5][4][14];
  int max_drift_tdc = -1;

private:
  // Private attributes
  const bool debug_;
};

#endif
