#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveSelection.hh"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "helper.hh"  // adjacent_cluster

#define NUM_CSC_CHAMBERS 6*9   // 18 in ME1; 9 in ME2,3,4; 9 from neighbor sector.
                               // Arranged in FW as 6 stations, 9 chambers per station
#define NUM_RPC_CHAMBERS 7*6   // 6 in RE1,2; 12 in RE3,4; 6 from neighbor sector.
                               // Arranged in FW as 7 stations, 6 chambers per station (unconfirmed!)

using CSCData = L1TMuon::TriggerPrimitive::CSCData;
using RPCData = L1TMuon::TriggerPrimitive::RPCData;


void PrimitiveSelection::configure(
      int verbose, int endcap, int sector, int bx,
      int bxShiftCSC, int bxShiftRPC,
      bool includeNeighbor, bool duplicateTheta,
      bool bugME11Dupes
) {
  verbose_ = verbose;
  endcap_  = endcap;
  sector_  = sector;
  bx_      = bx;

  bxShiftCSC_      = bxShiftCSC;
  bxShiftRPC_      = bxShiftRPC;

  includeNeighbor_ = includeNeighbor;
  duplicateTheta_  = duplicateTheta;
  bugME11Dupes_    = bugME11Dupes;
}

// Specialized for CSC
template<>
void PrimitiveSelection::process(
    CSCTag tag,
    const TriggerPrimitiveCollection& muon_primitives,
    std::map<int, TriggerPrimitiveCollection>& selected_csc_map
) const {
  TriggerPrimitiveCollection::const_iterator tp_it  = muon_primitives.begin();
  TriggerPrimitiveCollection::const_iterator tp_end = muon_primitives.end();

  for (; tp_it != tp_end; ++tp_it) {
    TriggerPrimitive new_tp = *tp_it;  // make a copy and apply patches to this copy

    // Patch the CLCT pattern number
    // It should be 0-10, see: L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.cc
    bool patchPattern = true;
    if (patchPattern) {
      if (new_tp.getCSCData().pattern == 11 || new_tp.getCSCData().pattern == 12) {  // 11, 12 -> 10
        new_tp.accessCSCData().pattern = 10;
      }
    }

    // Patch the LCT quality number
    // It should be 1-15, see: L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.cc
    bool patchQuality = true;
    if (patchQuality) {
      if (new_tp.getCSCData().quality == 0) {  // 0 -> 1
        new_tp.accessCSCData().quality = 1;
      }
    }

    int selected_csc = select_csc(new_tp); // Returns CSC "link" index (0 - 53)

    if (selected_csc >= 0) {
      assert(selected_csc < NUM_CSC_CHAMBERS);
      selected_csc_map[selected_csc].push_back(new_tp);
    }
  }

  // Duplicate CSC muon primitives
  // If there are 2 LCTs in the same chamber with (strip, wire) = (s1, w1) and (s2, w2)
  // make all combinations with (s1, w1), (s2, w1), (s1, w2), (s2, w2)
  if (duplicateTheta_) {
    std::map<int, TriggerPrimitiveCollection>::iterator map_tp_it  = selected_csc_map.begin();
    std::map<int, TriggerPrimitiveCollection>::iterator map_tp_end = selected_csc_map.end();

    for (; map_tp_it != map_tp_end; ++map_tp_it) {
      int selected = map_tp_it->first;
      TriggerPrimitiveCollection& tmp_primitives = map_tp_it->second;  // pass by reference
      assert(tmp_primitives.size() <= 2);  // at most 2

      if (tmp_primitives.size() == 2) {
        if (
            (tmp_primitives.at(0).getStrip() != tmp_primitives.at(1).getStrip()) &&
            (tmp_primitives.at(0).getWire() != tmp_primitives.at(1).getWire())
        ) {
          // Swap wire numbers
          TriggerPrimitive tp0 = tmp_primitives.at(0);  // (s1,w1)
          TriggerPrimitive tp1 = tmp_primitives.at(1);  // (s2,w2)
          uint16_t tmp_keywire        = tp0.accessCSCData().keywire;
          tp0.accessCSCData().keywire = tp1.accessCSCData().keywire;  // (s1,w2)
          tp1.accessCSCData().keywire = tmp_keywire;                  // (s2,w1)

          tmp_primitives.insert(tmp_primitives.begin()+1, tp1);  // (s2,w1) at 2nd pos
          tmp_primitives.insert(tmp_primitives.begin()+2, tp0);  // (s1,w2) at 3rd pos
        }

        const bool is_csc_me11 = (0 <= selected && selected <= 2) || (9 <= selected && selected <= 11) || (selected == 45);  // ME1/1 sub 1 or ME1/1 sub 2 or ME1/1 from neighbor

        if (bugME11Dupes_ && is_csc_me11) {
          // For ME1/1, always make 4 LCTs without checking strip & wire combination
          if (tmp_primitives.size() == 2) {
            // Swap wire numbers
            TriggerPrimitive tp0 = tmp_primitives.at(0);  // (s1,w1)
            TriggerPrimitive tp1 = tmp_primitives.at(1);  // (s2,w2)
            uint16_t tmp_keywire        = tp0.accessCSCData().keywire;
            tp0.accessCSCData().keywire = tp1.accessCSCData().keywire;  // (s1,w2)
            tp1.accessCSCData().keywire = tmp_keywire;                  // (s2,w1)

            tmp_primitives.insert(tmp_primitives.begin()+1, tp1);  // (s2,w1) at 2nd pos
            tmp_primitives.insert(tmp_primitives.begin()+2, tp0);  // (s1,w2) at 3rd pos
          }
          assert(tmp_primitives.size() == 1 || tmp_primitives.size() == 4);
        }

      }  // end if tmp_primitives.size() == 2
    }  // end loop over selected_csc_map
  }  // end if duplicate theta
}

// Specialized for RPC
template<>
void PrimitiveSelection::process(
    RPCTag tag,
    const TriggerPrimitiveCollection& muon_primitives,
    std::map<int, TriggerPrimitiveCollection>& selected_rpc_map
) const {
  TriggerPrimitiveCollection::const_iterator tp_it  = muon_primitives.begin();
  TriggerPrimitiveCollection::const_iterator tp_end = muon_primitives.end();

  for (; tp_it != tp_end; ++tp_it) {
    int selected_rpc = select_rpc(*tp_it);  // Returns RPC "link" index (0 - 41)

    if (selected_rpc >= 0) {
      assert(selected_rpc < NUM_RPC_CHAMBERS);
      selected_rpc_map[selected_rpc].push_back(*tp_it);
    }
  }

  // Cluster RPC digis
  bool do_clustering = true;
  if (do_clustering) {
    // Define operator to sort the trigger primitives prior to clustering
    // Here, the RPC trigger primitives are already contained inside the same
    // sector, subsector, station, ring, BX. So, only sort by roll, strip
    struct {
      typedef TriggerPrimitive value_type;
      bool operator()(const value_type& lhs, const value_type& rhs) const {
        bool cmp = (
            std::make_pair(lhs.detId<RPCDetId>().roll(), lhs.getRPCData().strip) <
            std::make_pair(rhs.detId<RPCDetId>().roll(), rhs.getRPCData().strip)
        );
        return cmp;
      }
    } rpc_digi_less;

    // Define operators for the nearest-neighbor clustering algorithm
    // If two digis are next to each other (check strip_hi on the 'left', and
    // strip_low on the 'right'), cluster them (increment strip_hi on the 'left')
    struct {
      typedef TriggerPrimitive value_type;
      bool operator()(const value_type& lhs, const value_type& rhs) const {
        bool cmp = (
            (lhs.detId<RPCDetId>().roll() == rhs.detId<RPCDetId>().roll()) &&
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

    // Define operator to apply cuts as in firmware: keep first 2 RPC clusters,
    // max cluster size = 3 strips.
    // According to Karol Bunkowski, for one chamber (so 3 eta rolls) only up
    // to 2 hits (cluster centres) are produced. First two 'first' clusters are
    // chosen, and only after the cut on the cluster size is applied. So if
    // there are 1 large cluster and 2 small clusters, it is possible that
    // one of the two small clusters is discarded first, and the large cluster
    // then is removed by the cluster size cut, leaving only one cluster.
    struct {
      typedef TriggerPrimitive value_type;
      bool operator()(const value_type& x) const {
        int sz = x.getRPCData().strip_hi - x.getRPCData().strip_low + 1;
        return sz > 3;
      }
    } rpc_digi_cluster_size_cut;


    // Loop over selected_rpc_map, do the clustering
    std::map<int, TriggerPrimitiveCollection>::iterator map_tp_it  = selected_rpc_map.begin();
    std::map<int, TriggerPrimitiveCollection>::iterator map_tp_end = selected_rpc_map.end();

    for (; map_tp_it != map_tp_end; ++map_tp_it) {
      //int selected = map_tp_it->first;
      TriggerPrimitiveCollection& tmp_primitives = map_tp_it->second;  // pass by reference

      // Sanity check
      for (const auto& x : tmp_primitives) {
        assert(x.getRPCData().strip == x.getRPCData().strip_low);
        assert(x.getRPCData().strip == x.getRPCData().strip_hi);
        //std::cout << ".. before: st: " << x.detId<RPCDetId>().station() << " ri: " << x.detId<RPCDetId>().ring() << " sub: " << x.detId<RPCDetId>().subsector()
        //    << " bx: " << x.getRPCData().bx << " layer: " << x.getRPCData().layer << " roll: " << x.detId<RPCDetId>().roll()
        //    << " strip: " << x.getRPCData().strip << " strip_low: " << x.getRPCData().strip_low << " strip_hi: " << x.getRPCData().strip_hi << std::endl;
      }

      // Cluster
      std::sort(tmp_primitives.begin(), tmp_primitives.end(), rpc_digi_less);
      tmp_primitives.erase(
          adjacent_cluster(tmp_primitives.begin(), tmp_primitives.end(), rpc_digi_adjacent, rpc_digi_cluster),
          tmp_primitives.end()
      );

      // Keep the first two clusters
      if (tmp_primitives.size() > 2)
        tmp_primitives.erase(tmp_primitives.begin()+2, tmp_primitives.end());

      // Apply cluster size cut
      tmp_primitives.erase(
          std::remove_if(tmp_primitives.begin(), tmp_primitives.end(), rpc_digi_cluster_size_cut),
          tmp_primitives.end()
      );

      // Sanity check
      for (const auto& x : tmp_primitives) {
        assert(x.getRPCData().strip_low <= x.getRPCData().strip_hi);
        //std::cout << ".. after : st: " << x.detId<RPCDetId>().station() << " ri: " << x.detId<RPCDetId>().ring() << " sub: " << x.detId<RPCDetId>().subsector()
        //    << " bx: " << x.getRPCData().bx << " layer: " << x.getRPCData().layer << " roll: " << x.detId<RPCDetId>().roll()
        //    << " strip: " << x.getRPCData().strip << " strip_low: " << x.getRPCData().strip_low << " strip_hi: " << x.getRPCData().strip_hi << std::endl;
      }

    }  // end loop over selected_rpc_map
  }  // end if do_clustering
}

// _____________________________________________________________________________
// CSC functions
int PrimitiveSelection::select_csc(const TriggerPrimitive& muon_primitive) const {
  int selected = -1;

  if (muon_primitive.subsystem() == TriggerPrimitive::kCSC) {
    const CSCDetId& tp_detId = muon_primitive.detId<CSCDetId>();
    const CSCData&  tp_data  = muon_primitive.getCSCData();

    int tp_endcap    = tp_detId.endcap();
    int tp_sector    = tp_detId.triggerSector();
    int tp_station   = tp_detId.station();
    int tp_ring      = tp_detId.ring();
    int tp_chamber   = tp_detId.chamber();

    int tp_bx        = tp_data.bx;
    int tp_csc_ID    = tp_data.cscID;

    assert(MIN_ENDCAP <= tp_endcap && tp_endcap <= MAX_ENDCAP);
    assert(MIN_TRIGSECTOR <= tp_sector && tp_sector <= MAX_TRIGSECTOR);
    assert(1 <= tp_station && tp_station <= 4);
    assert(1 <= tp_csc_ID && tp_csc_ID <= 9);
    assert(tp_data.strip < 160);
    //assert(tp_data.keywire < 112);
    assert(tp_data.keywire < 128);
    assert(tp_data.valid == true);
    assert(tp_data.pattern <= 10);
    assert(tp_data.quality > 0);

    // Check using ME1/1a --> ring 4 convention
    if (tp_station == 1 && tp_ring == 1) {
      assert(tp_data.strip < 128);
      assert(1 <= tp_csc_ID && tp_csc_ID <= 3);
    }
    if (tp_station == 1 && tp_ring == 4) {
      assert(tp_data.strip < 128);
      assert(1 <= tp_csc_ID && tp_csc_ID <= 3);
    }

    // station 1 --> subsector 1 or 2
    // station 2,3,4 --> subsector 0
    int tp_subsector = (tp_station != 1) ? 0 : ((tp_chamber%6 > 2) ? 1 : 2);

    // Selection
    if (is_in_bx_csc(tp_bx)) {
      if (is_in_sector_csc(tp_endcap, tp_sector)) {
        selected = get_index_csc(tp_subsector, tp_station, tp_csc_ID, false);

      } else if (is_in_neighbor_sector_csc(tp_endcap, tp_sector, tp_subsector, tp_station, tp_csc_ID)) {
        selected = get_index_csc(tp_subsector, tp_station, tp_csc_ID, true);
      }
    }
  }

  return selected;
}

bool PrimitiveSelection::is_in_sector_csc(int tp_endcap, int tp_sector) const {
  return ((endcap_ == tp_endcap) && (sector_ == tp_sector));
}

bool PrimitiveSelection::is_in_neighbor_sector_csc(int tp_endcap, int tp_sector, int tp_subsector, int tp_station, int tp_csc_ID) const {
  auto get_neighbor = [](int sector) {
    return (sector == 1) ? 6 : sector - 1;
  };

  if (includeNeighbor_) {
    if ((endcap_ == tp_endcap) && (get_neighbor(sector_) == tp_sector)) {
      if (tp_station == 1) {
        if ((tp_subsector == 2) && (tp_csc_ID == 3 || tp_csc_ID == 6 || tp_csc_ID == 9))
          return true;

      } else {
        if (tp_csc_ID == 3 || tp_csc_ID == 9)
          return true;
      }
    }
  }
  return false;
}

bool PrimitiveSelection::is_in_bx_csc(int tp_bx) const {
  tp_bx += bxShiftCSC_;
  return (bx_ == tp_bx);
}

// Returns CSC input "link".  Index used by FW for unique chamber identification.
int PrimitiveSelection::get_index_csc(int tp_subsector, int tp_station, int tp_csc_ID, bool is_neighbor) const {
  int selected = -1;

  if (!is_neighbor) {
    if (tp_station == 1) {  // ME1: 0 - 8, 9 - 17
      selected = (tp_subsector-1) * 9 + (tp_csc_ID-1);
    } else {                // ME2,3,4: 18 - 26, 27 - 35, 36 - 44
      selected = (tp_station) * 9 + (tp_csc_ID-1);
    }

  } else {
    if (tp_station == 1) {  // ME1: 45 - 47
      selected = (5) * 9 + (tp_csc_ID-1)/3;
    } else {                // ME2,3,4: 48 - 53
      selected = (5) * 9 + (tp_station) * 2 - 1 + (tp_csc_ID-1 < 3 ? 0 : 1);
    }
  }

  return selected;
}

// _____________________________________________________________________________
// RPC functions
int PrimitiveSelection::select_rpc(const TriggerPrimitive& muon_primitive) const {
  int selected = -1;

  if (muon_primitive.subsystem() == TriggerPrimitive::kRPC) {
    const RPCDetId& tp_detId = muon_primitive.detId<RPCDetId>();
    const RPCData&  tp_data  = muon_primitive.getRPCData();

    int tp_region    = tp_detId.region();     // 0 for Barrel, +/-1 for +/- Endcap
    int tp_endcap    = (tp_region == -1) ? 2 : tp_region;
    int tp_sector    = tp_detId.sector();     // 1 - 6 (60 degrees in phi, sector 1 begins at -5 deg)
    int tp_subsector = tp_detId.subsector();  // 1 - 6 (10 degrees in phi; staggered in z)
    int tp_station   = tp_detId.station();    // 1 - 4
    int tp_ring      = tp_detId.ring();       // 2 - 3 (increasing theta)
    int tp_roll      = tp_detId.roll();       // 1 - 3 (decreasing theta; aka A - C; space between rolls is 9 - 15 in theta_fp)

    int tp_bx        = tp_data.bx;
    int tp_strip     = tp_data.strip;

    assert(tp_region != 0);
    assert(MIN_ENDCAP <= tp_endcap && tp_endcap <= MAX_ENDCAP);
    assert(MIN_TRIGSECTOR <= tp_sector && tp_sector <= MAX_TRIGSECTOR);
    assert(1 <= tp_subsector && tp_subsector <= 6);
    assert(1 <= tp_station && tp_station <= 4);
    assert(2 <= tp_ring && tp_ring <= 3);
    assert(1 <= tp_roll && tp_roll <= 3);
    assert(1 <= tp_strip && tp_strip <= 32);
    assert(tp_station > 2 || tp_ring != 3);  // stations 1 and 2 do not receive RPCs from ring 3

    // Selection
    if (is_in_bx_rpc(tp_bx)) {
      if (is_in_sector_rpc(tp_endcap, tp_sector, tp_subsector)) {
        selected = get_index_rpc(tp_station, tp_ring, tp_subsector, false);

      } else if (is_in_neighbor_sector_rpc(tp_endcap, tp_sector, tp_subsector)) {
        selected = get_index_rpc(tp_station, tp_ring, tp_subsector, true);
      }
    }
  }

  return selected;
}

bool PrimitiveSelection::is_in_sector_rpc(int tp_endcap, int tp_sector, int tp_subsector) const {
  // RPC sector X, subsectors 1-2 corresponds to CSC sector X-1
  // RPC sector X, subsectors 3-6 corresponds to CSC sector X
  auto get_real_sector = [](int sector, int subsector) {
    int corr = (subsector < 3) ? (sector == 1 ? +5 : -1) : 0;
    return sector + corr;
  };
  return ((endcap_ == tp_endcap) && (sector_ == get_real_sector(tp_sector, tp_subsector)));
}

bool PrimitiveSelection::is_in_neighbor_sector_rpc(int tp_endcap, int tp_sector, int tp_subsector) const {
  return (includeNeighbor_ && (endcap_ == tp_endcap) && (sector_ == tp_sector) && (tp_subsector == 2));
}

bool PrimitiveSelection::is_in_bx_rpc(int tp_bx) const {
  tp_bx += bxShiftRPC_;
  return (bx_ == tp_bx);
}

int PrimitiveSelection::get_index_rpc(int tp_station, int tp_ring, int tp_subsector, bool is_neighbor) const {
  int selected = -1;

  if (!is_neighbor) {
    if (tp_station <= 2) {  // RE1:  0 -  5, RE2:  6 - 11
      selected = ((tp_station - 1)*6) + ((tp_subsector + 3) % 6);
    } else {                // RE3: 12 - 23, RE4: 24 - 35
      selected = (2)*6 + ((tp_station - 3)*12) + ((tp_ring - 2)*6) + ((tp_subsector + 3) % 6);
    }

  } else {
    if (tp_station <= 2) {  // RE1: 36       RE2: 37
      selected = (2)*6 + (2)*12 + (tp_station - 1);
    } else {                // RE3: 38 - 39, RE4: 40 - 41
      selected = (2)*6 + (2)*12 + (1)*2 + ((tp_station - 3)*2) + (tp_ring - 2);
    }
  }

  return selected;
}
