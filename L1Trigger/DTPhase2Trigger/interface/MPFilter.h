#ifndef Phase2L1Trigger_DTTrigger_MPFilter_cc
#define Phase2L1Trigger_DTTrigger_MPFilter_cc

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

#include "L1Trigger/DTPhase2Trigger/interface/muonpath.h"
#include "L1Trigger/DTPhase2Trigger/interface/analtypedefs.h"
#include "L1Trigger/DTPhase2Trigger/interface/constants.h"
#include "L1Trigger/DTPhase2Trigger/interface/MPFilter.h"

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

class MPFilter {
 public:
  // Constructors and destructor
  MPFilter(const edm::ParameterSet& pset);
  virtual ~MPFilter();
  
  // Main methods
  virtual void initialise(const edm::EventSetup& iEventSetup)=0;
  virtual void run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, std::vector<metaPrimitive> &inMPath, std::vector<metaPrimitive> &outMPath)=0;
  virtual void run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, std::vector<MuonPath*> &inMPath, std::vector<MuonPath*> &outMPath)=0;
  
  virtual void finish()=0;
    
  // Other public methods
  
 private:

  
  // Private attributes
  Bool_t debug;
};


#endif
