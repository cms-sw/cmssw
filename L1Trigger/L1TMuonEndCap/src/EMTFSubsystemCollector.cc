#include "L1Trigger/L1TMuonEndCap/interface/EMTFSubsystemCollector.h"
#include "DataFormats/L1TMuon/interface/L1TMuonSubsystems.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"  // needed to handle RPCRecHit

#include "helper.h"  // adjacent_cluster

// _____________________________________________________________________________
// Specialized for DT
template <>
void EMTFSubsystemCollector::extractPrimitives(emtf::DTTag tag,
                                               const GeometryTranslator* tp_geom,
                                               const edm::Event& iEvent,
                                               const edm::EDGetToken& token1,
                                               const edm::EDGetToken& token2,
                                               TriggerPrimitiveCollection& out) const {
  edm::Handle<emtf::DTTag::digi_collection> phiContainer;
  iEvent.getByToken(token1, phiContainer);

  edm::Handle<emtf::DTTag::theta_digi_collection> thetaContainer;
  iEvent.getByToken(token2, thetaContainer);

  TriggerPrimitiveCollection muon_primitives;

  // Adapted from L1Trigger/L1TMuonBarrel/src/L1TMuonBarrelKalmanStubProcessor.cc
  constexpr int minPhiQuality = 0;
  constexpr int minBX = -3;
  constexpr int maxBX = 3;

  for (int bx = minBX; bx <= maxBX; bx++) {
    for (int wheel = -2; wheel <= 2; wheel++) {
      for (int sector = 0; sector < 12; sector++) {
        for (int station = 1; station < 5; station++) {
          if (wheel == -1 || wheel == 0 || wheel == 1)
            continue;  // do not include wheels -1, 0, +1
          if (station == 4)
            continue;  // do not include MB4

          // According to Michalis, in legacy BMTF, the second stub was coming as BXNUM=-1.
          // This is a code convention now, but you need bx-1 to get the proper second stub.
          emtf::DTTag::theta_digi_type const* theta_segm = thetaContainer->chThetaSegm(wheel, station, sector, bx);
          emtf::DTTag::digi_type const* phi_segm_high = phiContainer->chPhiSegm1(wheel, station, sector, bx);
          emtf::DTTag::digi_type const* phi_segm_low = phiContainer->chPhiSegm2(wheel, station, sector, bx - 1);

          // Find theta BTI group(s)
          bool has_theta_segm = false;
          int bti_group1 = -1;
          int bti_group2 = -1;

          // Case with theta segment
          if (theta_segm != nullptr) {
            has_theta_segm = true;

            for (unsigned int i = 0; i < 7; ++i) {
              if (theta_segm->position(i) != 0) {
                if (bti_group1 < 0) {
                  bti_group1 = i;
                  bti_group2 = i;
                } else {
                  bti_group2 = i;
                }
              }
            }
            emtf_assert(bti_group1 != -1 && bti_group2 != -1);
          }

          // 1st phi segment
          if (phi_segm_high != nullptr) {
            if (phi_segm_high->code() >= minPhiQuality) {
              DTChamberId detid(phi_segm_high->whNum(), phi_segm_high->stNum(), phi_segm_high->scNum() + 1);
              if (has_theta_segm) {
                muon_primitives.emplace_back(detid, *phi_segm_high, *theta_segm, bti_group1);
              } else {
                muon_primitives.emplace_back(detid, *phi_segm_high, 1);
              }
            }
          }

          // 2nd phi segment
          if (phi_segm_low != nullptr) {
            if (phi_segm_low->code() >= minPhiQuality) {
              DTChamberId detid(phi_segm_low->whNum(), phi_segm_low->stNum(), phi_segm_low->scNum() + 1);
              if (has_theta_segm) {
                muon_primitives.emplace_back(detid, *phi_segm_low, *theta_segm, bti_group2);
              } else {
                muon_primitives.emplace_back(detid, *phi_segm_low, 2);
              }
            }
          }

          // Duplicate DT muon primitives, if more than one theta segment, but only one phi segment
          if (phi_segm_high != nullptr && phi_segm_low == nullptr && bti_group1 != bti_group2) {
            DTChamberId detid(phi_segm_high->whNum(), phi_segm_high->stNum(), phi_segm_high->scNum() + 1);
            muon_primitives.emplace_back(detid, *phi_segm_high, *theta_segm, bti_group2);
          }

        }  // end loop over station
      }    // end loop over sector
    }      // end loop over wheel
  }        // end loop over bx

  // Remove duplicates using erase-remove idiom,
  // assuming the vector is already sorted
  muon_primitives.erase(std::unique(muon_primitives.begin(), muon_primitives.end()), muon_primitives.end());
  std::copy(muon_primitives.begin(), muon_primitives.end(), std::back_inserter(out));
  return;
}

// _____________________________________________________________________________
// Specialized for CSC
template <>
void EMTFSubsystemCollector::extractPrimitives(emtf::CSCTag tag,
                                               const GeometryTranslator* tp_geom,
                                               const edm::Event& iEvent,
                                               const edm::EDGetToken& token,
                                               TriggerPrimitiveCollection& out) const {
  edm::Handle<emtf::CSCTag::digi_collection> cscDigis;
  iEvent.getByToken(token, cscDigis);

  auto chamber = cscDigis->begin();
  auto chend = cscDigis->end();
  for (; chamber != chend; ++chamber) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for (; digi != dend; ++digi) {
      out.emplace_back((*chamber).first, *digi);
    }
  }
  return;
}

// _____________________________________________________________________________
// Specialized for RPC
template <>
void EMTFSubsystemCollector::extractPrimitives(emtf::RPCTag tag,
                                               const GeometryTranslator* tp_geom,
                                               const edm::Event& iEvent,
                                               const edm::EDGetToken& token,
                                               TriggerPrimitiveCollection& out) const {
  edm::Handle<emtf::RPCTag::digi_collection> rpcDigis;
  iEvent.getByToken(token, rpcDigis);

  TriggerPrimitiveCollection muon_primitives;

  auto chamber = rpcDigis->begin();
  auto chend = rpcDigis->end();
  for (; chamber != chend; ++chamber) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for (; digi != dend; ++digi) {
      if ((*chamber).first.region() != 0) {  // 0 is barrel
        if ((*chamber).first.station() <= 2 && (*chamber).first.ring() == 3)
          continue;  // do not include RE1/3, RE2/3
        if ((*chamber).first.station() >= 3 && (*chamber).first.ring() == 1)
          continue;  // do not include RE3/1, RE4/1 (iRPC)

        muon_primitives.emplace_back((*chamber).first, *digi);
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

// Specialized for RPC (using RecHits)
template <>
void EMTFSubsystemCollector::extractPrimitives(emtf::RPCTag tag,
                                               const GeometryTranslator* tp_geom,
                                               const edm::Event& iEvent,
                                               const edm::EDGetToken& token1,
                                               const edm::EDGetToken& token2,
                                               TriggerPrimitiveCollection& out) const {
  constexpr int maxClusterSize = 3;

  //edm::Handle<RPCTag::digi_collection> rpcDigis;
  //iEvent.getByToken(token1, rpcDigis);

  edm::Handle<emtf::RPCTag::rechit_collection> rpcRecHits;
  iEvent.getByToken(token2, rpcRecHits);

  auto rechit = rpcRecHits->begin();
  auto rhend = rpcRecHits->end();
  for (; rechit != rhend; ++rechit) {
    const RPCDetId& detid = rechit->rpcId();
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(tp_geom->getRPCGeometry().roll(detid));
    if (roll == nullptr)
      continue;

    if (detid.region() != 0) {  // 0 is barrel
      if (detid.station() <= 2 && detid.ring() == 3)
        continue;  // do not include RE1/3, RE2/3
      if (detid.station() >= 3 && detid.ring() == 1)
        continue;  // do not include RE3/1, RE4/1 (iRPC)

      if (rechit->clusterSize() <= maxClusterSize) {
        out.emplace_back(detid, *rechit);
      }
    }
  }
  return;
}

// _____________________________________________________________________________
// Specialized for iRPC
template <>
void EMTFSubsystemCollector::extractPrimitives(emtf::IRPCTag tag,
                                               const GeometryTranslator* tp_geom,
                                               const edm::Event& iEvent,
                                               const edm::EDGetToken& token,
                                               TriggerPrimitiveCollection& out) const {
  edm::Handle<emtf::IRPCTag::digi_collection> irpcDigis;
  iEvent.getByToken(token, irpcDigis);

  TriggerPrimitiveCollection muon_primitives;

  auto chamber = irpcDigis->begin();
  auto chend = irpcDigis->end();
  for (; chamber != chend; ++chamber) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for (; digi != dend; ++digi) {
      if ((*chamber).first.region() != 0) {  // 0 is barrel
        if (!((*chamber).first.station() >= 3 && (*chamber).first.ring() == 1))
          continue;  // only RE3/1, RE4/1 (iRPC)

        muon_primitives.emplace_back((*chamber).first, *digi);
      }
    }
  }

  // Cluster the iRPC digis
  TriggerPrimitiveCollection clus_muon_primitives;
  cluster_rpc(muon_primitives, clus_muon_primitives);

  // Output
  std::copy(clus_muon_primitives.begin(), clus_muon_primitives.end(), std::back_inserter(out));
  return;
}

// Specialized for iRPC (using RecHits)
template <>
void EMTFSubsystemCollector::extractPrimitives(emtf::IRPCTag tag,
                                               const GeometryTranslator* tp_geom,
                                               const edm::Event& iEvent,
                                               const edm::EDGetToken& token1,
                                               const edm::EDGetToken& token2,
                                               TriggerPrimitiveCollection& out) const {
  constexpr int maxClusterSize = 6;

  //edm::Handle<emtf::IRPCTag::digi_collection> irpcDigis;
  //iEvent.getByToken(token1, irpcDigis);

  edm::Handle<emtf::IRPCTag::rechit_collection> irpcRecHits;
  iEvent.getByToken(token2, irpcRecHits);

  auto rechit = irpcRecHits->begin();
  auto rhend = irpcRecHits->end();
  for (; rechit != rhend; ++rechit) {
    const RPCDetId& detid = rechit->rpcId();
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(tp_geom->getRPCGeometry().roll(detid));
    if (roll == nullptr)
      continue;

    if (detid.region() != 0) {  // 0 is barrel
      if (!(detid.station() >= 3 && detid.ring() == 1))
        continue;  // only RE3/1, RE4/1 (iRPC)

      if (rechit->clusterSize() <= maxClusterSize) {
        out.emplace_back(detid, *rechit);
      }
    }
  }
  return;
}

// _____________________________________________________________________________
// Specialized for CPPF
template <>
void EMTFSubsystemCollector::extractPrimitives(emtf::CPPFTag tag,
                                               const GeometryTranslator* tp_geom,
                                               const edm::Event& iEvent,
                                               const edm::EDGetToken& token,
                                               TriggerPrimitiveCollection& out) const {
  edm::Handle<emtf::CPPFTag::digi_collection> cppfDigis;
  iEvent.getByToken(token, cppfDigis);

  for (const auto& digi : *cppfDigis) {
    out.emplace_back(digi.rpcId(), digi);
  }
  return;
}

// _____________________________________________________________________________
// Specialized for GEM
template <>
void EMTFSubsystemCollector::extractPrimitives(emtf::GEMTag tag,
                                               const GeometryTranslator* tp_geom,
                                               const edm::Event& iEvent,
                                               const edm::EDGetToken& token,
                                               TriggerPrimitiveCollection& out) const {
  edm::Handle<emtf::GEMTag::digi_collection> gemDigis;
  iEvent.getByToken(token, gemDigis);

  TriggerPrimitiveCollection muon_primitives;

  auto chamber = gemDigis->begin();
  auto chend = gemDigis->end();
  for (; chamber != chend; ++chamber) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for (; digi != dend; ++digi) {
      auto detid = (*chamber).first;
      // temporarily ignore 16-partition GE2/1 clusters, because the EMTF
      // is not yet adapted to handle these objects
      if (detid.isGE21() and digi->nPartitions() == GEMPadDigi::GE21SplitStrip)
        continue;
      muon_primitives.emplace_back((*chamber).first, *digi);
    }
  }

  // Make GEM coincidence pads
  TriggerPrimitiveCollection copad_muon_primitives;
  make_copad_gem(muon_primitives, copad_muon_primitives);

  // Output
  std::copy(copad_muon_primitives.begin(), copad_muon_primitives.end(), std::back_inserter(out));
  return;
}

// _____________________________________________________________________________
// Specialized for ME0
template <>
void EMTFSubsystemCollector::extractPrimitives(emtf::ME0Tag tag,
                                               const GeometryTranslator* tp_geom,
                                               const edm::Event& iEvent,
                                               const edm::EDGetToken& token,
                                               TriggerPrimitiveCollection& out) const {
  edm::Handle<emtf::ME0Tag::digi_collection> me0Digis;
  iEvent.getByToken(token, me0Digis);

  auto chamber = me0Digis->begin();
  auto chend = me0Digis->end();
  for (; chamber != chend; ++chamber) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for (; digi != dend; ++digi) {
      out.emplace_back((*chamber).first, *digi);
    }
  }
  return;
}

// _____________________________________________________________________________
// RPC functions
void EMTFSubsystemCollector::cluster_rpc(const TriggerPrimitiveCollection& muon_primitives,
                                         TriggerPrimitiveCollection& clus_muon_primitives) const {
  // Define operator to select RPC digis
  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& x) const { return (x.subsystem() == L1TMuon::kRPC); }
  } rpc_digi_select;

  // Define operator to sort the RPC digis prior to clustering.
  // Use rawId, bx and strip as the sorting id. RPC rawId fully specifies
  // sector, subsector, endcap, station, ring, layer, roll. Strip is used as
  // the least significant sorting id.
  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      bool cmp = (std::make_pair(std::make_pair(lhs.rawId(), lhs.getRPCData().bx), lhs.getRPCData().strip) <
                  std::make_pair(std::make_pair(rhs.rawId(), rhs.getRPCData().bx), rhs.getRPCData().strip));
      return cmp;
    }
  } rpc_digi_less;

  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      bool cmp = (std::make_pair(std::make_pair(lhs.rawId(), lhs.getRPCData().bx), lhs.getRPCData().strip) ==
                  std::make_pair(std::make_pair(rhs.rawId(), rhs.getRPCData().bx), rhs.getRPCData().strip));
      return cmp;
    }
  } rpc_digi_equal;

  // Define operators for the nearest-neighbor clustering algorithm.
  // If two digis are next to each other (check strip_hi on the 'left', and
  // strip_low on the 'right'), cluster them (increment strip_hi on the 'left')
  struct {
    typedef TriggerPrimitive value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      bool cmp = ((lhs.rawId() == rhs.rawId()) && (lhs.getRPCData().bx == rhs.getRPCData().bx) &&
                  (lhs.getRPCData().strip_hi + 1 == rhs.getRPCData().strip_low));
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
  clus_muon_primitives.clear();
  std::copy_if(
      muon_primitives.begin(), muon_primitives.end(), std::back_inserter(clus_muon_primitives), rpc_digi_select);

  // 2. Sort
  std::stable_sort(clus_muon_primitives.begin(), clus_muon_primitives.end(), rpc_digi_less);

  // 3. Remove duplicates
  clus_muon_primitives.erase(std::unique(clus_muon_primitives.begin(), clus_muon_primitives.end(), rpc_digi_equal),
                             clus_muon_primitives.end());

  // 4. Cluster adjacent digis
  clus_muon_primitives.erase(
      adjacent_cluster(clus_muon_primitives.begin(), clus_muon_primitives.end(), rpc_digi_adjacent, rpc_digi_cluster),
      clus_muon_primitives.end());
}

// _____________________________________________________________________________
// GEM functions
void EMTFSubsystemCollector::make_copad_gem(const TriggerPrimitiveCollection& muon_primitives,
                                            TriggerPrimitiveCollection& copad_muon_primitives) const {
  // Use the inner layer (layer 1) hit coordinates as output, and the outer
  // layer (layer 2) as coincidence

  // Adapted from L1Trigger/CSCTriggerPrimitives/src/GEMCoPadProcessor.cc
  constexpr unsigned int maxDeltaBX = 1;
  constexpr unsigned int maxDeltaRoll = 1;
  constexpr unsigned int maxDeltaPadGE11 = 3;  // it was 2
  constexpr unsigned int maxDeltaPadGE21 = 2;

  // Make sure that the difference is calculated using signed integer, and
  // output the absolute difference (as unsigned integer)
  auto calculate_delta = [](int a, int b) -> unsigned int { return std::abs(a - b); };

  // Create maps of GEM pads (key = detid), split by layer
  std::map<uint32_t, TriggerPrimitiveCollection> in_pads_layer1, in_pads_layer2;

  auto tp_it = muon_primitives.begin();
  auto tp_end = muon_primitives.end();

  for (; tp_it != tp_end; ++tp_it) {
    GEMDetId detid = tp_it->detId<GEMDetId>();
    emtf_assert(detid.layer() == 1 || detid.layer() == 2);
    emtf_assert(1 <= detid.roll() && detid.roll() <= 8);
    uint32_t layer = detid.layer();

    // Remove layer number and roll number from detid
    detid = GEMDetId(detid.region(), detid.ring(), detid.station(), 0, detid.chamber(), 0);

    if (layer == 1) {
      in_pads_layer1[detid.rawId()].push_back(*tp_it);
    } else {
      in_pads_layer2[detid.rawId()].push_back(*tp_it);
    }
  }

  // Build coincidences
  copad_muon_primitives.clear();

  auto map_tp_it = in_pads_layer1.begin();
  auto map_tp_end = in_pads_layer1.end();

  for (; map_tp_it != map_tp_end; ++map_tp_it) {
    const GEMDetId& detid = map_tp_it->first;
    const TriggerPrimitiveCollection& pads = map_tp_it->second;

    // find all corresponding ids with layer 2
    auto found = in_pads_layer2.find(detid);

    // empty range = no possible coincidence pads
    if (found == in_pads_layer2.end())
      continue;

    // now let's correlate the pads in two layers of this partition
    const TriggerPrimitiveCollection& co_pads = found->second;
    for (auto p = pads.begin(); p != pads.end(); ++p) {
      bool has_copad = false;
      int bend = 999999;

      for (auto co_p = co_pads.begin(); co_p != co_pads.end(); ++co_p) {
        unsigned int deltaPad = calculate_delta(p->getGEMData().pad, co_p->getGEMData().pad);
        unsigned int deltaBX = calculate_delta(p->getGEMData().bx, co_p->getGEMData().bx);
        unsigned int deltaRoll = calculate_delta(p->detId<GEMDetId>().roll(), co_p->detId<GEMDetId>().roll());

        // check the match in pad
        if ((detid.station() == 1 && deltaPad > maxDeltaPadGE11) ||
            (detid.station() == 2 && deltaPad > maxDeltaPadGE21))
          continue;

        // check the match in BX
        if (deltaBX > maxDeltaBX)
          continue;

        // check the match in roll
        if (deltaRoll > maxDeltaRoll)
          continue;

        has_copad = true;

        // recover the bend sign
        if (static_cast<unsigned int>(std::abs(bend)) > deltaPad) {
          if (co_p->getGEMData().pad >= p->getGEMData().pad)
            bend = deltaPad;
          else
            bend = -deltaPad;
        }
      }  // end loop over co_pads

      // Need to flip the bend sign depending on the parity
      bool isEven = (detid.chamber() % 2 == 0);
      if (!isEven) {
        bend = -bend;
      }

      // make a new coincidence pad digi
      if (has_copad) {
        copad_muon_primitives.push_back(*p);
      }
    }  // end loop over pads
  }    // end loop over in_pads_layer1
}
