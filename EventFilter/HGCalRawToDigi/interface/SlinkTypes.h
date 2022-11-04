#ifndef EventFilter_HGCalRawToDigi_SlinkTypes_h
#define EventFilter_HGCalRawToDigi_SlinkTypes_h

#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

namespace hgcal::econd {
  /// Event index (L1A/BX/orbit)
  typedef std::tuple<uint32_t, uint32_t, uint32_t> EventId;
  typedef std::tuple<uint8_t, uint8_t> ERx_t;
  struct ERxData {
    std::vector<uint16_t> adc, adcm, toa, tot;
    std::vector<uint8_t> tctp;
    uint32_t cm0, cm1;
  };
  typedef std::map<ERx_t, ERxData> ERxEvent;
  typedef std::map<EventId, ERxEvent> ECONDInputs;
  typedef std::pair<EventId, ERxEvent> ECONDEvent;
}  // namespace hgcal::econd

#endif
