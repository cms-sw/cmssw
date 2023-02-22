#include "EventFilter/HGCalRawToDigi/interface/HGCalECONDEmulator.h"

namespace hgcal::econd {
  ECONDEvent TrivialEmulator::next() {
    EventId evt_id{event_id_++, bx_id_++, orbit_id_++};
    ERxEvent evt;
    for (const auto& erx_id : params_.enabled_erxs) {
      ERx_t id{erx_id /*chip*/, 0 /*half*/};
      ERxData dummy_data{.adc = std::vector<uint16_t>(params_.num_channels_per_erx, 0),
                         .adcm = std::vector<uint16_t>(params_.num_channels_per_erx, 0),
                         .toa = std::vector<uint16_t>(params_.num_channels_per_erx, 0),
                         .tot = std::vector<uint16_t>(params_.num_channels_per_erx, 0),
                         .tctp = std::vector<uint8_t>(params_.num_channels_per_erx, 3),
                         .cm0 = 0,
                         .cm1 = 0};
      evt[id] = dummy_data;
    }
    return ECONDEvent{evt_id, evt};
  }
}  // namespace hgcal::econd
