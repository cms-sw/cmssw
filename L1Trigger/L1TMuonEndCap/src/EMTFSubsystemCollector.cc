#include "L1Trigger/L1TMuonEndCap/interface/EMTFSubsystemCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"


// Specialized for CSC
template<>
void EMTFSubsystemCollector::extractPrimitives(
    CSCTag tag, // Defined in interface/EMTFSubsystemTag.h, maps to CSCCorrelatedLCTDigi
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
    RPCTag tag, // Defined in interface/EMTFSubsystemTag.h, maps to RPCDigi
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

// Specialized for GEM
template<>
void EMTFSubsystemCollector::extractPrimitives(
    GEMTag tag, // Defined in interface/EMTFSubsystemTag.h, maps to GEMPadDigi
    const edm::Event& iEvent,
    const edm::EDGetToken& token,
    TriggerPrimitiveCollection& out
) {
  edm::Handle<GEMTag::digi_collection> gemDigis;
  iEvent.getByToken(token, gemDigis);

  auto chamber = gemDigis->begin();
  auto chend   = gemDigis->end();
  for( ; chamber != chend; ++chamber ) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for( ; digi != dend; ++digi ) {
      out.emplace_back((*chamber).first,*digi);
    }
  }
  return;
}
