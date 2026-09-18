#ifndef Phase2DAQFormatSpecification_H
#define Phase2DAQFormatSpecification_H

#include "DataFormats/FEDRawData/interface/SLinkRocketHeaders.h"

/**
 * @namespace Phase2DAQFormatSpecification
 * @brief Defines bit-level format specifications and constants for Phase-2 Outer Tracker DAQ.
 * 
 * The specifications in this namespace are used by the DAQ data formatting and 
 * unpacking modules to correctly interpret the raw data coming from the backend electronics
 * of the Phase-2 Outer Tracker system.
 *
 * @see DataFormats/FEDRawData/interface/SLinkRocketHeaders.h for S-Link protocol details.
 */

namespace Phase2DAQFormatSpecification {

  static const int C_NUM_BITS_BOARD_TYPE = 8;
  static const int C_NUM_BITS_BOARD_TYPE_INV = 8;
  static const int C_NUM_BITS_VERSION_MAJOR = 3;
  static const int C_NUM_BITS_VERSION_MINOR = 5;
  static const int C_NUM_BITS_MODE = 3;
  static const int C_NUM_BITS_ED = 1;
  static const int C_NUM_BITS_BOARD_ID = 8;
  static const int C_NUM_BITS_CORE_ID = 4;
  static const int C_NUM_BITS_RESERVED_TRAILER = 24;

  static const int DTC_HEADER_OT_PS = 0xC5;
  static const int DTC_HEADER_OT_PS_INV = 0x5C;
  static const int DTC_HEADER_OT_2S = 0xC4;
  static const int DTC_HEADER_OT_2S_INV = 0x4C;

  static const int DTC_HEADER_OFFSET = 0; // location where DTC HEADER Starts
  static const int DTC_TRAILER_OFFSET = 26;

  static const int DTC_CHANNEL_MASK_OFFSET = 22;
  static const int DTC_CHANNEL_MASK_SIZE = 2;

  static const int SLINK_HEADER_SIZE = sizeof(SLinkRocketHeader_v3) * 8 / 32;
  static const int SLINK_TRAILER_SIZE = sizeof(SLinkRocketTrailer_v3) * 8 / 32;
  static const int DTC_MASK_PROFILE_SIZE = 2; // in 32bit words

  // This CMSSW Version should be compatible against the versioining variables below.
  static const int VERSION_MAJOR_V1_0 = 0x1;
  static const int VERSION_MINOR_V1_0 = 0x0;
  
  static const int DTC_DAQ_HEADER = 0xFFFFFFFF;
  static const int N_BITS_PER_WORD = 32;
  static const int N_BYTES_PER_WORD = 4;
  static const int N_BYTES_PER_DTH_BINARY_WORD = 16;

  // Channel Header Information (Payload)
  static const int L1ID_MAX_VALUE = 0x1FF;
  static const int L1ID_BITS = 9;
  static const int CIC_ERROR_BITS = 9;
  static const int N_PIXEL_CLUSTER_BITS = 7;
  static const int N_STRIP_CLUSTER_BITS = 7;

  static const int CHIP_ID_MAX_VALUE = 0x7;
  static const int CHIP_ID_BITS = 3;

  static const int SCLUSTER_ADDRESS_2S_MAX_VALUE = 0x7F;
  static const int SCLUSTER_ADDRESS_BITS_2S = 8;
  // last 1 bit of sclusteraddress is used to discriminate seeding/corr, so effectively
  // the address is a 7 bit object
  static const int SCLUSTER_ADDRESS_ONLY_BITS_2S = 7;

  static const int SCLUSTER_ADDRESS_PS_MAX_VALUE = 0x7F;
  static const int SCLUSTER_ADDRESS_BITS_PS = 7;

  static const int SCLUSTER_ADDRESS_MASK = 0x7F;
  static const int IS_SEED_SENSOR_MASK = 0x01;

  static const int WIDTH_MAX_VALUE = 0x7;
  static const int WIDTH_BITS = 3;

  static const int MIP_BITS = 1;
  static const int MIP_BITS_MASK = 0x1;
  static const int PS_Z_BITS_MASK = 0xF;

  static const int SS_CLUSTER_BITS = 14;
  static const int PX_CLUSTER_BITS = 17;
  static const int Z_MAX_VALUE = 0;

  static const int CMSSW_TRACKER_ID = 0;

  static const int HEADER_N_LINES = 4;  // number of 32b lines of the tracker header
  static const int TRAILER_N_LINES = 4; // number of 32b lines of the tracker trailer
  static const int OFFSET_BITS = 16;    // length of the offset word
  static const int RESERVED_N_LINES = 2;

  static const int CIC_ERROR_MASK = 0x1FF;
  static const int N_CLUSTER_MASK = 0x7F;
  static const int SS_CLUSTER_WORD_MASK = 0x3FFF;
  static const int PX_CLUSTER_WORD_MASK = 0x1FFFF;

  typedef std::bitset<32> Word32Bits;

};  // namespace Phase2DAQFormatSpecification

#endif