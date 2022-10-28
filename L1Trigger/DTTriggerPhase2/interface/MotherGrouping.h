#ifndef Phase2L1Trigger_DTTrigger_MotherGrouping_h
#define Phase2L1Trigger_DTTrigger_MotherGrouping_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

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

class MotherGrouping {
public:
  // Constructors and destructor
  MotherGrouping(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  virtual ~MotherGrouping();

  // Main methods
  virtual void initialise(const edm::EventSetup& iEventSetup);
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   const DTDigiCollection& digis,
                   MuonPathPtrs& outMpath);
  virtual void finish();

  // Other public methods

  // Public attributes

private:
  // Private methods

  // Private attributes
  const bool debug_;
};

#endif
