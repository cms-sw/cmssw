#ifndef L1TMuonEndCap_SectorProcessorLUT_h
#define L1TMuonEndCap_SectorProcessorLUT_h

#include <cstdint>
#include <string>
#include <vector>


class SectorProcessorLUT {
public:
  explicit SectorProcessorLUT();
  ~SectorProcessorLUT();

  void read(bool pc_lut_data, int pc_lut_version);

  uint32_t get_ph_init(int fw_endcap, int fw_sector, int pc_lut_id) const;

  uint32_t get_ph_disp(int fw_endcap, int fw_sector, int pc_lut_id) const;

  uint32_t get_th_init(int fw_endcap, int fw_sector, int pc_lut_id) const;

  uint32_t get_th_disp(int fw_endcap, int fw_sector, int pc_lut_id) const;

  uint32_t get_th_lut(int fw_endcap, int fw_sector, int pc_lut_id, int pc_wire_id) const;

  uint32_t get_th_corr_lut(int fw_endcap, int fw_sector, int pc_lut_id, int pc_wire_strip_id) const;

  uint32_t get_ph_patt_corr(int pattern) const;

  uint32_t get_ph_patt_corr_sign(int pattern) const;

  uint32_t get_ph_zone_offset(int pc_station, int pc_chamber) const;

  uint32_t get_ph_init_hard(int fw_station, int fw_cscid) const;

  uint32_t get_cppf_lut_id(int rpc_region, int rpc_sector, int rpc_station, int rpc_ring, int rpc_subsector, int rpc_roll) const;

  uint32_t get_cppf_ph_lut(int rpc_region, int rpc_sector, int rpc_station, int rpc_ring, int rpc_subsector, int rpc_roll, int halfstrip, bool is_neighbor) const;

  uint32_t get_cppf_th_lut(int rpc_region, int rpc_sector, int rpc_station, int rpc_ring, int rpc_subsector, int rpc_roll) const;

private:
  void read_file(const std::string& filename, std::vector<uint32_t>& vec);

  void read_cppf_file(const std::string& filename, std::vector<uint32_t>& vec1, std::vector<uint32_t>& vec2, bool local);

  std::vector<uint32_t> ph_init_neighbor_;
  std::vector<uint32_t> ph_disp_neighbor_;
  std::vector<uint32_t> th_init_neighbor_;
  std::vector<uint32_t> th_disp_neighbor_;
  std::vector<uint32_t> th_lut_neighbor_;
  std::vector<uint32_t> th_corr_lut_neighbor_;

  std::vector<uint32_t> ph_patt_corr_;
  std::vector<uint32_t> ph_patt_corr_sign_;
  std::vector<uint32_t> ph_zone_offset_;
  std::vector<uint32_t> ph_init_hard_;

  std::vector<uint32_t> cppf_ph_lut_;
  std::vector<uint32_t> cppf_th_lut_;

  int version_;  // init: 0xFFFFFFFF
};

#endif
