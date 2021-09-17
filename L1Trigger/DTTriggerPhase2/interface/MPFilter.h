#ifndef Phase2L1Trigger_DTTrigger_MPFilter_h
#define Phase2L1Trigger_DTTrigger_MPFilter_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

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
                   MuonPathPtrs& inMPath,
                   MuonPathPtrs& outMPath) = 0;

  virtual void finish() = 0;

  // Other public methods

private:
  // Private attributes
  bool debug_;
};

#endif
