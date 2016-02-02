// 
// Class: L1TMuonAssignedInternalTrackProducer
//
// Info: A configurable track processor where the pt assignment/refinement 
//       algorithms are controlled by config files.
//       The closest representation of this in the old hardware is 
//       the pt lookup table
//
// Author: L. Gray (FNAL)
//

#include <memory>
#include <map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "L1Trigger/L1TMuonEndCap/interface/MuonTriggerPrimitive.h"
#include "L1Trigger/L1TMuonEndCap/interface/MuonTriggerPrimitiveFwd.h"

#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrack.h"
#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrackFwd.h"

// interface to Pt assignment and refinement
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentUnitFactory.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtRefinementUnitFactory.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace L1TMuon;

typedef edm::ParameterSet PSet;
typedef std::unique_ptr<PtAssignmentUnit> pPAU;
typedef std::unique_ptr<PtRefinementUnit> pPRU;

class L1TMuonAssignedInternalTrackProducer : public edm::EDProducer {    
public:
  L1TMuonAssignedInternalTrackProducer(const PSet&);
  ~L1TMuonAssignedInternalTrackProducer() {}

  void produce(edm::Event&, const edm::EventSetup&);  
private:
  pPAU _ptassign;
  pPRU _ptrefine;
};

L1TMuonAssignedInternalTrackProducer::L1TMuonAssignedInternalTrackProducer(const PSet& p) {  
  // configure and build pt assignment unit
  std::unique_ptr<PtAssignmentUnitFactory> 
    fPAU(PtAssignmentUnitFactory::get());
  if( p.existsAs<PSet>("PtAssignmentUnit") ) {
    PSet PAU_config = p.getParameterSet("PtAssignmentUnit");
    std::string PAU_type = p.getParameter<std::string>("type");
    _ptassign.reset( fPAU->create( PAU_type,
				   PAU_config) );
  } else {
    _ptassign.reset(NULL);
  }
  // configure and build pt refinement unit
  std::unique_ptr<PtRefinementUnitFactory> 
    fPRU(PtRefinementUnitFactory::get());
  if( p.existsAs<PSet>("PtRefinementUnit") ) {
    PSet PRU_config = p.getParameterSet("PtRefinementUnit");
    std::string PRU_type = p.getParameter<std::string>("type");
    _ptrefine.reset( fPRU->create( PRU_type,
				   PRU_config) );
  } else {
    _ptrefine.reset(NULL);
  }
  
  fPAU.release();
  fPRU.release();
}

void L1TMuonAssignedInternalTrackProducer::produce(edm::Event& ev, 
					   const edm::EventSetup& es) {  
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonAssignedInternalTrackProducer);
