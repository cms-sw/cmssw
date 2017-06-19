#ifndef L1TMuonEndCap_PrimitiveSelection_h
#define L1TMuonEndCap_PrimitiveSelection_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"


class PrimitiveSelection {
public:
  void configure(
      int verbose, int endcap, int sector, int bx,
      int bxShiftCSC, int bxShiftRPC, int bxShiftGEM,
      bool includeNeighbor, bool duplicateTheta,
      bool bugME11Dupes
  );

  template<typename T>
  void process(
      T tag,
      const TriggerPrimitiveCollection& muon_primitives,
      std::map<int, TriggerPrimitiveCollection>& selected_prim_map
  ) const;

  // Put the hits from CSC, RPC, GEM together in one collection
  void merge(
      std::map<int, TriggerPrimitiveCollection>& selected_csc_map,
      std::map<int, TriggerPrimitiveCollection>& selected_rpc_map,
      std::map<int, TriggerPrimitiveCollection>& selected_gem_map,
      std::map<int, TriggerPrimitiveCollection>& selected_prim_map
  ) const;

  // CSC functions
  // If selected, return an index 0-53, else return -1
  // The index 0-53 roughly corresponds to an input link. It maps to the
  // 2D index [station][chamber] used in the firmware, with size [5:0][8:0].
  // Station 5 = neighbor sector, all stations.
  int select_csc(const TriggerPrimitive& muon_primitive) const;

  bool is_in_sector_csc(int tp_endcap, int tp_sector) const;

  bool is_in_neighbor_sector_csc(int tp_endcap, int tp_sector, int tp_subsector, int tp_station, int tp_csc_ID) const;

  bool is_in_bx_csc(int tp_bx) const;

  int get_index_csc(int tp_subsector, int tp_station, int tp_csc_ID, bool is_neighbor) const;

  // RPC functions
  void cluster_rpc(const TriggerPrimitiveCollection& muon_primitives, TriggerPrimitiveCollection& clus_muon_primitives) const;

  int select_rpc(const TriggerPrimitive& muon_primitive) const;

  bool is_in_sector_rpc(int tp_endcap, int tp_sector, int tp_subsector) const;

  bool is_in_neighbor_sector_rpc(int tp_endcap, int tp_sector, int tp_subsector) const;

  bool is_in_bx_rpc(int tp_bx) const;

  int get_index_rpc(int tp_station, int tp_ring, int tp_subsector, bool is_neighbor) const;

  // GEM functions
  void cluster_gem(const TriggerPrimitiveCollection& muon_primitives, TriggerPrimitiveCollection& clus_muon_primitives) const;

  int select_gem(const TriggerPrimitive& muon_primitive) const;

  bool is_in_sector_gem(int tp_endcap, int tp_sector) const;

  bool is_in_neighbor_sector_gem(int tp_endcap, int tp_sector, int tp_subsector, int tp_station, int tp_csc_ID) const;

  bool is_in_bx_gem(int tp_bx) const;

  int get_index_gem(int tp_subsector, int tp_station, int tp_csc_ID, bool is_neighbor) const;


private:
  int verbose_, endcap_, sector_, bx_;

  int bxShiftCSC_, bxShiftRPC_, bxShiftGEM_;

  bool includeNeighbor_, duplicateTheta_;

  bool bugME11Dupes_;
};

#endif
