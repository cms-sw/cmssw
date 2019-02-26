#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessorLUT.h"

#include <cassert>
#include <iostream>
#include <fstream>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


SectorProcessorLUT::SectorProcessorLUT() :
    version_(0xFFFFFFFF)
{

}

SectorProcessorLUT::~SectorProcessorLUT() {

}

void SectorProcessorLUT::read(bool pc_lut_data, int pc_lut_version) {
  if (version_ == pc_lut_version)  return;

  edm::LogInfo("L1T") << "EMTF using pc_lut_ver: " << pc_lut_version
		      << ", configured for " << (pc_lut_data ? "data" : "MC");

  std::string coord_lut_dir = "";
  if      (pc_lut_version == 0)
    coord_lut_dir = "ph_lut_v1";       // All year 2016
  else if (pc_lut_version == 1)
    coord_lut_dir = "ph_lut_v2";       // Beginning of 2017, improved alignment from ideal CMS geometry (MC)
  else if (pc_lut_version == 2 && pc_lut_data)
    coord_lut_dir = "ph_lut_v3_data";  // Update in September 2017 from ReReco alignment, data only
  else if (pc_lut_version == 2)
    coord_lut_dir = "ph_lut_v2";       // MC still uses ideal CMS aligment
  else if (pc_lut_version == -1 && pc_lut_data)
    coord_lut_dir = "ph_lut_v3_data";  // September 2017 data LCT alignment, but use local CPPF LUTs for RPC
  else if (pc_lut_version == -1)
    coord_lut_dir = "ph_lut_v2";       // MC using ideal CMS LCT alignment, but use local CPPF LUTs for RPC
  else
    throw cms::Exception("SectorProcessorLUT")
      << "Trying to use EMTF pc_lut_version = " << pc_lut_version << ", does not exist!";
  // Will catch user trying to run with Global Tag settings on 2016 data, rather than fakeEmtfParams. - AWB 08.06.17

  std::string coord_lut_path = "L1Trigger/L1TMuon/data/emtf_luts/" + coord_lut_dir + "/";

  read_file(coord_lut_path+"ph_init_neighbor.txt",     ph_init_neighbor_);
  read_file(coord_lut_path+"ph_disp_neighbor.txt",     ph_disp_neighbor_);
  read_file(coord_lut_path+"th_init_neighbor.txt",     th_init_neighbor_);
  read_file(coord_lut_path+"th_disp_neighbor.txt",     th_disp_neighbor_);
  read_file(coord_lut_path+"th_lut_neighbor.txt",      th_lut_neighbor_);
  read_file(coord_lut_path+"th_corr_lut_neighbor.txt", th_corr_lut_neighbor_);

  std::string cppf_coord_lut_path = "L1Trigger/L1TMuon/data/cppf/";  // Coordinate LUTs actually used by CPPF
  bool use_local_cppf_files = (pc_lut_version == -1);
  if (use_local_cppf_files) {  // More accurate coordinate transformation LUTs from Jia Fu
    cppf_coord_lut_path = "L1Trigger/L1TMuon/data/cppf_luts/angleScale_v1/";
  }

  read_cppf_file(cppf_coord_lut_path, cppf_ph_lut_, cppf_th_lut_, use_local_cppf_files);  // cppf filenames are hardcoded in the function

  if (ph_init_neighbor_.size() != 2*6*61) {  // [endcap_2][sector_6][chamber_61]
    throw cms::Exception("SectorProcessorLUT")
        << "Expected ph_init_neighbor_ to get " << 2*6*61 << " values, "
        << "got " << ph_init_neighbor_.size() << " values.";
  }

  if (ph_disp_neighbor_.size() != 2*6*61) {  // [endcap_2][sector_6][chamber_61]
    throw cms::Exception("SectorProcessorLUT")
        << "Expected ph_disp_neighbor_ to get " << 2*6*61 << " values, "
        << "got " << ph_disp_neighbor_.size() << " values.";
  }

  if (th_init_neighbor_.size() != 2*6*61) {  // [endcap_2][sector_6][chamber_61]
    throw cms::Exception("SectorProcessorLUT")
        << "Expected th_init_neighbor_ to get " << 2*6*61 << " values, "
        << "got " << th_init_neighbor_.size() << " values.";
  }

  if (th_disp_neighbor_.size() != 2*6*61) {  // [endcap_2][sector_6][chamber_61]
    throw cms::Exception("SectorProcessorLUT")
        << "Expected th_disp_neighbor_ to get " << 2*6*61 << " values, "
        << "got " << th_disp_neighbor_.size() << " values.";
  }

  if (th_lut_neighbor_.size() != 2*6*61*128) {  // [endcap_2][sector_6][chamber_61][wire_128]
    throw cms::Exception("SectorProcessorLUT")
        << "Expected th_lut_neighbor_ to get " << 2*6*61*128 << " values, "
        << "got " << th_lut_neighbor_.size() << " values.";
  }

  if (th_corr_lut_neighbor_.size() != 2*6*7*128) {  // [endcap_2][sector_6][chamber_61][strip_wire_128]
    throw cms::Exception("SectorProcessorLUT")
        << "Expected th_corr_lut_neighbor_ to get " << 2*6*7*128 << " values, "
        << "got " << th_corr_lut_neighbor_.size() << " values.";
  }

  if (cppf_ph_lut_.size() != 2*6*6*6*3*64) {  // [endcap_2][rpc_sector_6][rpc_station_ring_6][rpc_subsector_6][rpc_roll_3][rpc_halfstrip_64]
    throw cms::Exception("SectorProcessorLUT")
        << "Expected cppf_ph_lut_ to get " << 2*6*6*6*3*64 << " values, "
        << "got " << cppf_ph_lut_.size() << " values.";
  }

  if (cppf_th_lut_.size() != 2*6*6*6*3) {  // [endcap_2][rpc_sector_6][rpc_station_ring_6][rpc_subsector_6][rpc_roll_3]
    throw cms::Exception("SectorProcessorLUT")
        << "Expected cppf_th_lut_ to get " << 2*6*6*6*3 << " values, "
        << "got " << cppf_th_lut_.size() << " values.";
  }

  // clct pattern convertion array from CMSSW
  //{0.0, 0.0, -0.60,  0.60, -0.64,  0.64, -0.23,  0.23, -0.21,  0.21, 0.0}
  // 0    0    -5      +5    -5      +5    -2      +2    -2      +2    0
  ph_patt_corr_ = {
    0, 0, 5, 5, 5, 5, 2, 2, 2, 2, 0
  };
  if (ph_patt_corr_.size() != 11) {
    throw cms::Exception("SectorProcessorLUT")
        << "Expected ph_patt_corr_ to get " << 11 << " values, "
        << "got " << ph_patt_corr_.size() << " values.";
  }

  ph_patt_corr_sign_ = {
    0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0
  };
  if (ph_patt_corr_sign_.size() != 11) {
    throw cms::Exception("SectorProcessorLUT")
        << "Expected ph_patt_corr_sign_ to get " << 11 << " values, "
        << "got " << ph_patt_corr_sign_.size() << " values.";
  }

  ph_zone_offset_ = {
    39,57,76,39,58,76,41,60,79,
    95,114,132,95,114,133,98,116,135,
    38,76,113,39,58,76,95,114,132,
    38,76,113,39,58,76,95,114,132,
    38,76,113,38,57,76,95,113,132,
    21,21,23,1,21,1,21,1,20
  };
  if (ph_zone_offset_.size() != 6*9) {
    throw cms::Exception("SectorProcessorLUT")
        << "Expected ph_zone_offset_ to get " << 6*9 << " values, "
        << "got " << ph_zone_offset_.size() << " values.";
  }

  // start phi of each chamber in reduced precision, for zone hits,
  // with negative offset to allow for possible chamber movement
  ph_init_hard_ = {
    39,  57,  76, 39,  58,  76, 41,  60,  79, 39,  57,  76, 21, 21, 23, 21,
    95, 114, 132, 95, 114, 133, 98, 116, 135, 95, 114, 132,  0,  0,  0,  0,
    38,  76, 113, 39,  58,  76, 95, 114, 132,  1,  21,   0,  0,  0,  0,  0,
    38,  76, 113, 39,  58,  76, 95, 114, 132,  1,  21,   0,  0,  0,  0,  0,
    38,  76, 113, 38,  57,  76, 95, 113, 132,  1,  20,   0,  0,  0,  0,  0
  };
  if (ph_init_hard_.size() != 5*16) {
    throw cms::Exception("SectorProcessorLUT")
        << "Expected ph_init_hard_ to get " << 5*16 << " values, "
        << "got " << ph_init_hard_.size() << " values.";
  }

  version_ = pc_lut_version;
  return;
}

uint32_t SectorProcessorLUT::get_ph_init(int fw_endcap, int fw_sector, int pc_lut_id) const {
  size_t index = (fw_endcap * 6 + fw_sector) * 61 + pc_lut_id;
  return ph_init_neighbor_.at(index);
}

uint32_t SectorProcessorLUT::get_ph_disp(int fw_endcap, int fw_sector, int pc_lut_id) const {
  size_t index = (fw_endcap * 6 + fw_sector) * 61 + pc_lut_id;
  return ph_disp_neighbor_.at(index);
}

uint32_t SectorProcessorLUT::get_th_init(int fw_endcap, int fw_sector, int pc_lut_id) const {
  size_t index = (fw_endcap * 6 + fw_sector) * 61 + pc_lut_id;
  return th_init_neighbor_.at(index);
}

uint32_t SectorProcessorLUT::get_th_disp(int fw_endcap, int fw_sector, int pc_lut_id) const {
  size_t index = (fw_endcap * 6 + fw_sector) * 61 + pc_lut_id;
  return th_disp_neighbor_.at(index);
}

uint32_t SectorProcessorLUT::get_th_lut(int fw_endcap, int fw_sector, int pc_lut_id, int pc_wire_id) const {
  int pc_lut_id2 = pc_lut_id;

  // Make ME1/1a the same as ME1/1b
  if ((9 <= pc_lut_id2 && pc_lut_id2 < 12) || (25 <= pc_lut_id2 && pc_lut_id2 < 28))
    pc_lut_id2 -= 9;
  // Make ME1/1a neighbor the same as ME1/1b
  if (pc_lut_id2 == 15)
    pc_lut_id2 -= 3;

  size_t index = ((fw_endcap * 6 + fw_sector) * 61 + pc_lut_id2) * 128 + pc_wire_id;
  return th_lut_neighbor_.at(index);
}

uint32_t SectorProcessorLUT::get_th_corr_lut(int fw_endcap, int fw_sector, int pc_lut_id, int pc_wire_strip_id) const {
  int pc_lut_id2 = pc_lut_id;

  // Make ME1/1a the same as ME1/1b
  if ((9 <= pc_lut_id2 && pc_lut_id2 < 12) || (25 <= pc_lut_id2 && pc_lut_id2 < 28))
    pc_lut_id2 -= 9;
  // Make ME1/1a neighbor the same as ME1/1b
  if (pc_lut_id2 == 15)
    pc_lut_id2 -= 3;

  if (pc_lut_id2 <= 3) {
    pc_lut_id2 -= 0;
  } else if (pc_lut_id2 == 12) {
    pc_lut_id2 -= 9;
  } else if (16 <= pc_lut_id2 && pc_lut_id2 < 19) {
    pc_lut_id2 -= 12;
  } else {
    throw cms::Exception("SectorProcessorLUT")
      << "get_th_corr_lut(): out of range pc_lut_id: " << pc_lut_id;
  }

  size_t index = ((fw_endcap * 6 + fw_sector) * 7 + pc_lut_id2) * 128 + pc_wire_strip_id;
  return th_corr_lut_neighbor_.at(index);
}

uint32_t SectorProcessorLUT::get_ph_patt_corr(int pattern) const {
  return ph_patt_corr_.at(pattern);
}

uint32_t SectorProcessorLUT::get_ph_patt_corr_sign(int pattern) const {
  return ph_patt_corr_sign_.at(pattern);
}

uint32_t SectorProcessorLUT::get_ph_zone_offset(int pc_station, int pc_chamber) const {
  size_t index = pc_station * 9 + pc_chamber;
  return ph_zone_offset_.at(index);
}

uint32_t SectorProcessorLUT::get_ph_init_hard(int fw_station, int fw_cscid) const {
  size_t index = fw_station * 16 + fw_cscid;
  return ph_init_hard_.at(index);
}

uint32_t SectorProcessorLUT::get_cppf_lut_id(int rpc_region, int rpc_sector, int rpc_station, int rpc_ring, int rpc_subsector, int rpc_roll) const {
  uint32_t iendcap = (rpc_region == -1) ? 1 : 0;
  uint32_t isector = (rpc_sector - 1);
  uint32_t istationring = (rpc_station >= 3) ? ((rpc_station - 3) * 2 + (rpc_ring - 2) + 2) : (rpc_station - 1);
  uint32_t isubsector = (rpc_subsector - 1);
  uint32_t iroll = (rpc_roll - 1);
  return ((((iendcap * 6 + isector) * 6 + istationring) * 6 + isubsector) * 3 + iroll);
}

uint32_t SectorProcessorLUT::get_cppf_ph_lut(int rpc_region, int rpc_sector, int rpc_station, int rpc_ring, int rpc_subsector, int rpc_roll, int halfstrip, bool is_neighbor) const {
  size_t th_index       = get_cppf_lut_id(rpc_region, rpc_sector, rpc_station, rpc_ring, rpc_subsector, rpc_roll);
  size_t ph_index       = (th_index * 64) + (halfstrip - 1);
  uint32_t ph           = cppf_ph_lut_.at(ph_index);
  if (!is_neighbor && rpc_subsector == 2)
    ph += 900;
  return ph;
}

uint32_t SectorProcessorLUT::get_cppf_th_lut(int rpc_region, int rpc_sector, int rpc_station, int rpc_ring, int rpc_subsector, int rpc_roll) const {
  size_t th_index       = get_cppf_lut_id(rpc_region, rpc_sector, rpc_station, rpc_ring, rpc_subsector, rpc_roll);
  uint32_t th           = cppf_th_lut_.at(th_index);
  return th;
}

void SectorProcessorLUT::read_file(const std::string& filename, std::vector<uint32_t>& vec) {
  vec.clear();

  std::ifstream infile;
  infile.open(edm::FileInPath(filename).fullPath().c_str());

  int buf;
  while (infile >> buf) {
    buf = (buf == -999) ? 0 : buf;
    vec.push_back(buf);
  }
  infile.close();
}

void SectorProcessorLUT::read_cppf_file(const std::string& filename, std::vector<uint32_t>& vec1, std::vector<uint32_t>& vec2, bool local) {
  auto get_rpc_region = [](uint32_t id) { return (static_cast<int>((id >> 0) & 0X3) + (-1)); };
  auto get_rpc_sector = [](uint32_t id) { return (static_cast<int>((id >> 7) & 0XF) + (1)); };
  auto get_rpc_ring = [](uint32_t id) { return (static_cast<int>((id >> 2) & 0X7) + (1)); };
  auto get_rpc_station = [](uint32_t id) { return (static_cast<int>((id >> 5) & 0X3) + (1)); };
  auto get_rpc_subsector = [](uint32_t id) { return (static_cast<int>((id >> 12) & 0X7) + (1)); };
  auto get_rpc_roll = [](uint32_t id) { return (static_cast<int>((id >> 15) & 0X7) + (0)); };

  std::vector<std::string> cppf_filenames = {
    "angleScale_RPC_CPPFp1.txt",
    "angleScale_RPC_CPPFp2.txt",
    "angleScale_RPC_CPPFp3.txt",
    "angleScale_RPC_CPPFp4.txt",
    "angleScale_RPC_CPPFn1.txt",
    "angleScale_RPC_CPPFn2.txt",
    "angleScale_RPC_CPPFn3.txt",
    "angleScale_RPC_CPPFn4.txt",
  };


  vec1.clear();
  vec2.clear();
  vec1.resize(2*6*6*6*3*64, 0);
  vec2.resize(2*6*6*6*3, 0);

  for (size_t i = 0; i < cppf_filenames.size(); ++i) {
    std::ifstream infile;
    infile.open(edm::FileInPath(filename + cppf_filenames.at(i)).fullPath().c_str());

    // std::cout << "\n\nOpening CPPF LUT file " << cppf_filenames.at(i) << std::endl;

    int buf1, buf2, buf3, buf4, buf5, buf6;
    // Special variables for transforming centrally-provided CPPF LUTs
    int buf1_prev = 0, buf2_prev = 0, halfstrip_prev = 0; // Values from previous line in file
    int line_num = 0;   // Line number in file
    int count_dir = -1; // Direction of half-strip counting: +1 is up, -1 is down
    int dStrip = 0;     // Offset for half-strip from full strip
    while ((infile >> buf1) && (infile >> buf2) && (infile >> buf3) && (infile >> buf4) && (infile >> buf5) && (infile >> buf6)) {

      if ((line_num % 192) == 191) line_num += 1; // Gap in central files vs. Jia Fu's files
      line_num += 1;
      // On roughly every-other line, files in L1Trigger/L1TMuon/data/cppf have 0 in the first three columns
      // Skips a "0 0 0" line once every 192 lines
      if ((line_num % 2) == 1) {
	buf1_prev = buf1;
	buf2_prev = buf2;
      }

      if (local && (buf1 == 0 || buf2 == 0)) {
	throw cms::Exception("SectorProcessorLUT") << "Expected non-0 values, got buf1 = " << buf1 << ", buf2 = " << buf2;
      }
      if (!local && (buf1_prev == 0 || buf2_prev == 0)) {
	throw cms::Exception("SectorProcessorLUT") << "Expected non-0 values, got buf1_prev = " << buf1_prev << ", buf2_prev = " << buf2_prev;
      }

      uint32_t id           = (local ? buf1 : buf1_prev);
      int32_t rpc_region    = get_rpc_region(id);
      int32_t rpc_sector    = get_rpc_sector(id);
      int32_t rpc_station   = get_rpc_station(id);
      int32_t rpc_ring      = get_rpc_ring(id);
      int32_t rpc_subsector = get_rpc_subsector(id);
      int32_t rpc_roll      = get_rpc_roll(id);

      // Offset into halfstrips from centrally-provided LUTs
      if ( buf2_prev*2 > halfstrip_prev + 8 ||
	   buf2_prev*2 < halfstrip_prev - 8 ) { // Starting a new series of strips
	if (buf2_prev == 1) count_dir = +1; // Starting from a low number, counting up
	else                count_dir = -1; // Starting from a high number, counting down
      }
      if (count_dir == -1) dStrip = (buf2_prev*2 == halfstrip_prev     ? 1 : 0);
      if (count_dir == +1) dStrip = (buf2_prev*2 == halfstrip_prev + 2 ? 1 : 0);
      if (buf2_prev*2 < halfstrip_prev - 8 && buf2_prev == 1) dStrip = 1;

      //uint32_t strip        = buf2;
      uint32_t halfstrip    = (local ? buf2 : buf2_prev*2 - dStrip);  // I modified the local text files to use 'halfstrip' instead of 'strip' in column 2
      halfstrip_prev        = halfstrip;

      uint32_t ph           = buf5;
      uint32_t th           = buf6;

      size_t th_index       = get_cppf_lut_id(rpc_region, rpc_sector, rpc_station, rpc_ring, rpc_subsector, rpc_roll);
      size_t ph_index       = (th_index * 64) + (halfstrip - 1);

      // std::cout << id << " " << rpc_region << " " << rpc_sector << " " << rpc_station << " " << rpc_ring << " "
      // 		<< rpc_subsector << " " << rpc_roll << " " << halfstrip << " " << th_index << " " << ph_index << std::endl;

      vec1.at(ph_index) = ph;
      if (halfstrip == 1)
        vec2.at(th_index) = th;

      // Fill gap in centrally-provided LUTs once every 192 lines
      if (!local && (line_num % 192) == 191)
	vec1.at(ph_index+1) = ph;

    } // End while ((infile >> buf1) && ... && (infile >> buf6))
    infile.close();
  } // End loop over CPPF LUT files
}
