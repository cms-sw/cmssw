#ifndef Phase2L1Trigger_DTTrigger_LateralityProvider_h
#define Phase2L1Trigger_DTTrigger_LateralityProvider_h

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

using latcomb = std::vector<short>;
using lat_vector = std::vector<latcomb>;

class LateralityProvider {
public:
  // Constructors and destructor
  LateralityProvider(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  virtual ~LateralityProvider();

  // Main methods
  virtual void initialise(const edm::EventSetup& iEventSetup);
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   MuonPathPtrs& inMpath,
                   std::vector<lat_vector>& lateralities) = 0;

  virtual void finish();

  // Other public methods

  // Public attributes
  lat_vector LAT_VECTOR_NULL = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

private:
  // Private methods

  // Private attributes
  const bool debug_;
};

#endif
