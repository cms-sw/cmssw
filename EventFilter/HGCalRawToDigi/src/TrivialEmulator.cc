#include "EventFilter/HGCalRawToDigi/interface/HGCalECONDEmulator.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalRawDataDefinitions.h"

using namespace hgcal::econd;

ECONDInput TrivialEmulator::next() {
  EventId evt_id{event_id_++, bx_id_++, orbit_id_++};
  ERxInput evt;
  for (const auto& erx_id : params_.enabled_erxs) {
    ERxId_t id{erx_id /*chip*/, 0 /*half*/};
    ERxData dummy_data{
        .cm0 = 0x12,
        .cm1 = 0x34,
        .tctp = std::vector<ToTStatus>(params_.num_channels_per_erx, static_cast<ToTStatus>(params_.default_totstatus)),
        .adc = std::vector<uint16_t>(params_.num_channels_per_erx, 0),
        .adcm = std::vector<uint16_t>(params_.num_channels_per_erx, 0),
        .toa = std::vector<uint16_t>(params_.num_channels_per_erx, 0),
        .tot = std::vector<uint16_t>(params_.num_channels_per_erx, 0),
        .meta = std::vector<uint32_t>{}};
    evt[id] = dummy_data;
  }
  return ECONDInput{evt_id, evt};
}
