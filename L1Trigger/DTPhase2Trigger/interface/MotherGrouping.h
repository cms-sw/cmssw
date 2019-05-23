#ifndef Phase2L1Trigger_DTTrigger_MotherGrouping_cc
#define Phase2L1Trigger_DTTrigger_MotherGrouping_cc

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

#include "L1Trigger/DTPhase2Trigger/interface/MotherGrouping.h"


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

class MotherGrouping {
  public:
    // Constructors and destructor
    MotherGrouping(const edm::ParameterSet& pset);
    virtual ~MotherGrouping();
    
    // Main methods
    virtual void initialise(const edm::EventSetup& iEventSetup);
    virtual void run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, DTDigiCollection digis, std::vector<MuonPath*> *outMpath);
    virtual void finish();
    
    // Other public methods
    
    // Public attributes
    
  private:
    // Private methods
    
    // Private attributes
    Bool_t debug;
};


#endif
