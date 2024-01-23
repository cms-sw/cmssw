#ifndef Phase2L1Trigger_DTTrigger_MuonPathAnalyzer_h
#define Phase2L1Trigger_DTTrigger_MuonPathAnalyzer_h

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include "L1Trigger/DTTriggerPhase2/interface/GlobalCoordsObtainer.h"
#include "L1Trigger/DTTriggerPhase2/interface/LateralityProvider.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MuonPathAnalyzer {
public:
  // Constructors and destructor
  MuonPathAnalyzer(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  virtual ~MuonPathAnalyzer();

  // Main methods
  virtual void initialise(const edm::EventSetup& iEventSetup);
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   MuonPathPtrs& inMpath,
                   std::vector<cmsdt::metaPrimitive>& metaPrimitives) = 0;
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   MuonPathPtrs& inMpath,
                   std::vector<lat_vector>& lateralities,
                   std::vector<cmsdt::metaPrimitive>& metaPrimitives) = 0;
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   std::vector<cmsdt::metaPrimitive>& inMPaths,
                   std::vector<cmsdt::metaPrimitive>& outMPaths) = 0;
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   MuonPathPtrs& inMpath,
                   MuonPathPtrs& outMPath) = 0;

  virtual void finish();

  // Other public methods

  // Public attributes

private:
  // Private methods

  // Private attributes
  const bool debug_;
};

#endif
