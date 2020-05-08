#ifndef Phase2L1Trigger_DTTrigger_MuonPathAnalyzer_cc
#define Phase2L1Trigger_DTTrigger_MuonPathAnalyzer_cc

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"

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

class MuonPathAnalyzer {
public:
  // Constructors and destructor
  MuonPathAnalyzer(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  virtual ~MuonPathAnalyzer();

  // Main methods
  virtual void initialise(const edm::EventSetup& iEventSetup);
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   std::vector<MuonPath*>& inMpath,
                   std::vector<metaPrimitive>& metaPrimitives) = 0;
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   std::vector<MuonPath*>& inMpath,
                   std::vector<MuonPath*>& outMPath) = 0;

  virtual void finish();

  // Other public methods

  // Public attributes

private:
  // Private methods

  // Private attributes
  bool debug;
};

#endif
