#ifndef EventFilter_HGCalRawToDigi_SlinkTypes_h
#define EventFilter_HGCalRawToDigi_SlinkTypes_h

#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include "EventFilter/HGCalRawToDigi/interface/HGCalRawDataDefinitions.h"

namespace hgcal::econd {

  /// Event index (L1A/BX/orbit)
  typedef std::tuple<uint32_t, uint32_t, uint32_t> EventId;

  /// chip/half
  typedef std::pair<uint8_t, uint8_t> ERxId_t;

  /// parsed e-rx data
  struct ERxData {
    uint32_t cm0{0}, cm1{0};
    std::vector<ToTStatus> tctp;
    std::vector<uint16_t> adc, adcm, toa, tot;
    std::vector<uint32_t> meta;  ///< additional words accompanying the e-rx data
    uint32_t crc32{0};
  };

  /// eRx data maps
  typedef std::map<ERxId_t, ERxData> ERxInput;

  /// ECON-D inputs for a given event
  typedef std::pair<EventId, ERxInput> ECONDInput;

  /// collection of ECON-D inputs
  typedef std::map<EventId, ERxInput> ECONDInputColl;

  /// list of channels enabled in eRx
  typedef std::vector<bool> ERxChannelEnable;
}  // namespace hgcal::econd

#endif
