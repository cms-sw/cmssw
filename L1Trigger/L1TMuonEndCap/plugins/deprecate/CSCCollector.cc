#include "L1Trigger/L1TMuonEndCap/interface/CSCCollector.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
 
using namespace L1TMuon;

CSCCollector::CSCCollector( const edm::ParameterSet& ps ):
  SubsystemCollector(ps) {
}

void CSCCollector::
extractPrimitives(const edm::Event& ev, 
		  const edm::EventSetup& es, 
		  std::vector<TriggerPrimitive>& out) const {
  edm::Handle<CSCCorrelatedLCTDigiCollection> cscDigis;  
  ev.getByLabel(_src,cscDigis);    

  auto chamber = cscDigis->begin();
  auto chend  = cscDigis->end();
  for( ; chamber != chend; ++chamber ) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for( ; digi != dend; ++digi ) {
      out.push_back(TriggerPrimitive((*chamber).first,*digi));
    }
  }    
}

#include "L1Trigger/L1TMuonEndCap/interface/SubsystemCollectorFactory.h"
DEFINE_EDM_PLUGIN( SubsystemCollectorFactory, CSCCollector, "CSCCollector");
