#ifndef Phase2L1Trigger_DTTrigger_MPRedundantFilter_cc
#define Phase2L1Trigger_DTTrigger_MPRedundantFilter_cc

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

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
  virtual ~MPRedundantFilter();

  // Main methods
  void initialise(const edm::EventSetup& iEventSetup);
  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           std::vector<metaPrimitive>& inMPath,
           std::vector<metaPrimitive>& outMPath){};
  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           MuonPathPtrs& inMPath,
           MuonPathPtrs& outMPath);
  void finish() { buffer.clear(); };

  // Other public methods

private:
  void filter(MuonPathPtr& mpath, MuonPathPtrs& outMPaths);
  bool isInBuffer(MuonPathPtr& mpath);

  // Private attributes
  bool debug;
  unsigned int MaxBufferSize;
  std::deque<MuonPathPtr> buffer;
};

#endif
