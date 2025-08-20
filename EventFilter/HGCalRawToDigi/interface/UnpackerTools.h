#ifndef EventFilter_HGCalRawToDigi_interface_UnpackerTools_h
#define EventFilter_HGCalRawToDigi_interface_UnpackerTools_h
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"

#include <string>

namespace hgcal {

  /**
     @short this function re-computes the CRC of the ECON-D packet and compares it with the trailer of the packet
     false is returned if there is a mismatch
   */
  bool econdCRCAnalysis(const uint64_t *header, const uint32_t pos, const uint32_t payloadLength);

}  // namespace hgcal

#endif
