#include "L1Trigger/L1TMuonEndCap/interface/EMTFSubsystemCollector.hh"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"


// Specialized for CSC
template<>
void EMTFSubsystemCollector::extractPrimitives(
    CSCTag tag, // Defined in interface/EMTFSubsystemTag.hh, maps to CSCCorrelatedLCTDigi
    const edm::Event& iEvent,
    const edm::EDGetToken& token,
    TriggerPrimitiveCollection& out
) {
  edm::Handle<CSCTag::digi_collection> cscDigis;
  iEvent.getByToken(token, cscDigis);

  auto chamber = cscDigis->begin();
  auto chend   = cscDigis->end();
  for( ; chamber != chend; ++chamber ) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for( ; digi != dend; ++digi ) {
      // emplace_back does the same thing as push_back: appends to the end of the vector
      out.emplace_back((*chamber).first,*digi);
    }
  }
  return;
}

// Specialized for RPC
template<>
void EMTFSubsystemCollector::extractPrimitives(
    RPCTag tag, // Defined in interface/EMTFSubsystemTag.hh, maps to RPCDigi
    const edm::Event& iEvent,
    const edm::EDGetToken& token,
    TriggerPrimitiveCollection& out
) {
  edm::Handle<RPCTag::digi_collection> rpcDigis;
  iEvent.getByToken(token, rpcDigis);

  auto chamber = rpcDigis->begin();
  auto chend   = rpcDigis->end();
  for( ; chamber != chend; ++chamber ) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for( ; digi != dend; ++digi ) {
      if ((*chamber).first.region() != 0) {  // 0 is barrel
        if (!((*chamber).first.station() <= 2 && (*chamber).first.ring() == 3)) {  // do not include RE1/3, RE2/3
          out.emplace_back((*chamber).first,digi->strip(),(*chamber).first.layer(),digi->bx());
        }
      }
    }
  }
  return;
}
