#include "L1Trigger/L1TMuonEndCap/interface/EMTFSubsystemCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "helper.h"  // adjacent_cluster


// Specialized for CSC
template<>
void EMTFSubsystemCollector::extractPrimitives(
    CSCTag tag, // Defined in interface/EMTFSubsystemTag.h, maps to CSCCorrelatedLCTDigi
    const edm::Event& iEvent,
    const edm::EDGetToken& token,
    TriggerPrimitiveCollection& out
) const {
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
) const {
  edm::Handle<RPCTag::digi_collection> rpcDigis;
  iEvent.getByToken(token, rpcDigis);

  TriggerPrimitiveCollection muon_primitives;

  auto chamber = rpcDigis->begin();
  auto chend   = rpcDigis->end();
  for( ; chamber != chend; ++chamber ) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for( ; digi != dend; ++digi ) {
      if ((*chamber).first.region() != 0) {  // 0 is barrel
        if ((*chamber).first.station() <= 2 && (*chamber).first.ring() == 3)  continue;  // do not include RE1/3, RE2/3
        if ((*chamber).first.station() >= 3 && (*chamber).first.ring() == 1)  continue;  // do not include RE3/1, RE4/1

        muon_primitives.emplace_back((*chamber).first,*digi);
      }
    }
  }

  // Cluster the RPC digis
  TriggerPrimitiveCollection clus_muon_primitives;
  cluster_rpc(muon_primitives, clus_muon_primitives);

  // Output
  std::copy(clus_muon_primitives.begin(), clus_muon_primitives.end(), std::back_inserter(out));
  return;
}

// Specialized for GEM
template<>
void EMTFSubsystemCollector::extractPrimitives(
    GEMTag tag, // Defined in interface/EMTFSubsystemTag.h, maps to GEMPadDigi
    const edm::Event& iEvent,
    const edm::EDGetToken& token,
    TriggerPrimitiveCollection& out
) const {
  edm::Handle<GEMTag::digi_collection> gemDigis;
  iEvent.getByToken(token, gemDigis);

  TriggerPrimitiveCollection muon_primitives;

  auto chamber = gemDigis->begin();
  auto chend   = gemDigis->end();
  for( ; chamber != chend; ++chamber ) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for( ; digi != dend; ++digi ) {
      muon_primitives.emplace_back((*chamber).first,*digi);
    }
  }

  // Cluster the GEM digis.
  TriggerPrimitiveCollection copad_muon_primitives;
  make_copad_gem(muon_primitives, copad_muon_primitives);

  TriggerPrimitiveCollection clus_muon_primitives;
  cluster_gem(copad_muon_primitives, clus_muon_primitives);

  // Output
  std::copy(clus_muon_primitives.begin(), clus_muon_primitives.end(), std::back_inserter(out));
  return;
}


// _____________________________________________________________________________
// RPC functions
void EMTFSubsystemCollector::cluster_rpc(const TriggerPrimitiveCollection& muon_primitives, TriggerPrimitiveCollection& clus_muon_primitives) const {
  // Define operator to select RPC digis
  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& x) const {
      return (x.subsystem() == TriggerPrimitive::kRPC);
    }
  } rpc_digi_select;

  // Define operator to sort the RPC digis prior to clustering.
  // Use rawId, bx and strip as the sorting id. RPC rawId fully specifies
  // sector, subsector, endcap, station, ring, layer, roll. Strip is used as
  // the least significant sorting id.
  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      bool cmp = (
          std::make_pair(std::make_pair(lhs.rawId(), lhs.getRPCData().bx), lhs.getRPCData().strip) <
          std::make_pair(std::make_pair(rhs.rawId(), rhs.getRPCData().bx), rhs.getRPCData().strip)
      );
      return cmp;
    }
  } rpc_digi_less;

  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      bool cmp = (
          std::make_pair(std::make_pair(lhs.rawId(), lhs.getRPCData().bx), lhs.getRPCData().strip) ==
          std::make_pair(std::make_pair(rhs.rawId(), rhs.getRPCData().bx), rhs.getRPCData().strip)
      );
      return cmp;
    }
  } rpc_digi_equal;

  // Define operators for the nearest-neighbor clustering algorithm.
  // If two digis are next to each other (check strip_hi on the 'left', and
  // strip_low on the 'right'), cluster them (increment strip_hi on the 'left')
  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      bool cmp = (
          (lhs.rawId() == rhs.rawId()) &&
          (lhs.getRPCData().bx == rhs.getRPCData().bx) &&
          (lhs.getRPCData().strip_hi+1 == rhs.getRPCData().strip_low)
      );
      return cmp;
    }
  } rpc_digi_adjacent;

  struct {
    typedef TriggerPrimitive value_type;
    void operator()(value_type& lhs, value_type& rhs) {  // pass by reference
      lhs.accessRPCData().strip_hi += 1;
    }
  } rpc_digi_cluster;

  // ___________________________________________________________________________
  // Do clustering using C++ <algorithm> functions

  // 1. Select RPC digis
  std::copy_if(muon_primitives.begin(), muon_primitives.end(), std::back_inserter(clus_muon_primitives), rpc_digi_select);

  // 2. Sort
  std::stable_sort(clus_muon_primitives.begin(), clus_muon_primitives.end(), rpc_digi_less);

  // 3. Remove duplicates
  clus_muon_primitives.erase(
      std::unique(clus_muon_primitives.begin(), clus_muon_primitives.end(), rpc_digi_equal),
      clus_muon_primitives.end()
  );

  // 4. Cluster adjacent digis
  clus_muon_primitives.erase(
      adjacent_cluster(clus_muon_primitives.begin(), clus_muon_primitives.end(), rpc_digi_adjacent, rpc_digi_cluster),
      clus_muon_primitives.end()
  );
}


// _____________________________________________________________________________
// GEM functions
void EMTFSubsystemCollector::make_copad_gem(const TriggerPrimitiveCollection& muon_primitives, TriggerPrimitiveCollection& copad_muon_primitives) const {
  // Use the inner layer (layer 1) hit coordinates as output, and the outer
  // layer (layer 2) as coincidence
  // Copied from: L1Trigger/CSCTriggerPrimitives/src/GEMCoPadProcessor.cc

  const unsigned int maxDeltaBX = 1;
  const unsigned int maxDeltaPadGE11 = 2;
  const unsigned int maxDeltaPadGE21 = 2;

  std::map<int, TriggerPrimitiveCollection> in_pads_layer1, in_pads_layer2;

  TriggerPrimitiveCollection::const_iterator tp_it  = muon_primitives.begin();
  TriggerPrimitiveCollection::const_iterator tp_end = muon_primitives.end();

  for (; tp_it != tp_end; ++tp_it) {
    const TriggerPrimitive& muon_primitive = *tp_it;

    // Split by layer
    if (muon_primitive.subsystem() == TriggerPrimitive::kGEM) {
      const GEMDetId& tp_detId = muon_primitive.detId<GEMDetId>();
      assert(tp_detId.layer() == 1 || tp_detId.layer() == 2);
      if (tp_detId.layer() == 1) {
        in_pads_layer1[tp_detId.rawId()].push_back(muon_primitive);
      } else {
        in_pads_layer2[tp_detId.rawId()].push_back(muon_primitive);
      }

      // Modified copad logic
      bool modified_copad_logic = false;
      if (modified_copad_logic) {
        if (tp_detId.layer() == 1) {
          auto id = tp_detId;
          const GEMDetId co_detId(id.region(), id.ring(), id.station(), 2, id.chamber(), id.roll());
          const GEMPadDigi co_digi(muon_primitive.getGEMData().pad, muon_primitive.getGEMData().bx);
          const TriggerPrimitive co_muon_primitive(co_detId, co_digi);
          in_pads_layer2[co_detId.rawId()].push_back(co_muon_primitive);
        } else {
          auto id = tp_detId;
          const GEMDetId co_detId(id.region(), id.ring(), id.station(), 1, id.chamber(), id.roll());
          const GEMPadDigi co_digi(muon_primitive.getGEMData().pad, muon_primitive.getGEMData().bx);
          const TriggerPrimitive co_muon_primitive(co_detId, co_digi);
          in_pads_layer1[co_detId.rawId()].push_back(co_muon_primitive);
        }
      }
    }
  }

  std::map<int, TriggerPrimitiveCollection>::iterator map_tp_it  = in_pads_layer1.begin();
  std::map<int, TriggerPrimitiveCollection>::iterator map_tp_end = in_pads_layer1.end();

  for (; map_tp_it != map_tp_end; ++map_tp_it) {
    const GEMDetId& id = map_tp_it->first;
    const TriggerPrimitiveCollection& pads = map_tp_it->second;
    assert(id.layer() == 1);

    // find the corresponding id with layer=2 and same roll number
    const GEMDetId co_id(id.region(), id.ring(), id.station(), 2, id.chamber(), id.roll());

    // empty range = no possible coincidence pads
    auto found = in_pads_layer2.find(co_id);
    if (found == in_pads_layer2.end())  continue;

    // now let's correlate the pads in two layers of this partition
    const TriggerPrimitiveCollection& co_pads = found->second;
    for (TriggerPrimitiveCollection::const_iterator p = pads.begin(); p != pads.end(); ++p) {
      for (TriggerPrimitiveCollection::const_iterator co_p = co_pads.begin(); co_p != co_pads.end(); ++co_p) {
        unsigned int deltaPad = std::abs(p->getGEMData().pad - co_p->getGEMData().pad);
        unsigned int deltaBX = std::abs(p->getGEMData().bx - co_p->getGEMData().bx);

        // check the match in pad
        if ((id.station() == 1 && deltaPad > maxDeltaPadGE11) || (id.station() == 2 && deltaPad > maxDeltaPadGE21))
          continue;

        // check the match in BX
        if (deltaBX > maxDeltaBX)
          continue;

        // make a new coincidence pad digi
        copad_muon_primitives.push_back(*p);
      }
    }
  }
}

void EMTFSubsystemCollector::cluster_gem(const TriggerPrimitiveCollection& muon_primitives, TriggerPrimitiveCollection& clus_muon_primitives) const {
  // Define operator to select GEM digis
  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& x) const {
      return (x.subsystem() == TriggerPrimitive::kGEM);
    }
  } gem_digi_select;

  // Define operator to sort the GEM digis prior to clustering.
  // Use rawId, bx and pad as the sorting id. GEM rawId fully specifies
  // endcap, station, ring, layer, roll, chamber. Pad is used as
  // the least significant sorting id.
  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      bool cmp = (
          std::make_pair(std::make_pair(lhs.rawId(), lhs.getGEMData().bx), lhs.getGEMData().pad) <
          std::make_pair(std::make_pair(rhs.rawId(), rhs.getGEMData().bx), rhs.getGEMData().pad)
      );
      return cmp;
    }
  } gem_digi_less;

  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      bool cmp = (
          std::make_pair(std::make_pair(lhs.rawId(), lhs.getGEMData().bx), lhs.getGEMData().pad) ==
          std::make_pair(std::make_pair(rhs.rawId(), rhs.getGEMData().bx), rhs.getGEMData().pad)
      );
      return cmp;
    }
  } gem_digi_equal;

  // Define operators for the nearest-neighbor clustering algorithm.
  // If two digis are next to each other (check pad_hi on the 'left', and
  // pad_low on the 'right'), cluster them (increment pad_hi on the 'left')
  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      bool cmp = (
          (lhs.rawId() == rhs.rawId()) &&
          (lhs.getGEMData().bx == rhs.getGEMData().bx) &&
          (lhs.getGEMData().pad_hi+1 == rhs.getGEMData().pad_low)
      );
      return cmp;
    }
  } gem_digi_adjacent;

  struct {
    typedef TriggerPrimitive value_type;
    void operator()(value_type& lhs, value_type& rhs) {  // pass by reference
      lhs.accessGEMData().pad_hi += 1;
    }
  } gem_digi_cluster;

  // ___________________________________________________________________________
  // Do clustering using C++ <algorithm> functions

  // 1. Select GEM digis
  std::copy_if(muon_primitives.begin(), muon_primitives.end(), std::back_inserter(clus_muon_primitives), gem_digi_select);

  // 2. Sort
  std::stable_sort(clus_muon_primitives.begin(), clus_muon_primitives.end(), gem_digi_less);

  // 3. Remove duplicates
  clus_muon_primitives.erase(
      std::unique(clus_muon_primitives.begin(), clus_muon_primitives.end(), gem_digi_equal),
      clus_muon_primitives.end()
  );

  // 4. Cluster adjacent digis
  clus_muon_primitives.erase(
      adjacent_cluster(clus_muon_primitives.begin(), clus_muon_primitives.end(), gem_digi_adjacent, gem_digi_cluster),
      clus_muon_primitives.end()
  );
}
