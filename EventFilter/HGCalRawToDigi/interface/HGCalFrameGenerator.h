#ifndef EventFilter_HGCalRawToDigi_HGCalFrameGenerator_h
#define EventFilter_HGCalRawToDigi_HGCalFrameGenerator_h

#include "DataFormats/HGCalDigi/interface/HGCalRawDataEmulatorInfo.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalECONDEmulatorParameters.h"
#include "EventFilter/HGCalRawToDigi/interface/SlinkTypes.h"

#include <cstdint>
#include <vector>

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm
namespace CLHEP {
  class HepRandomEngine;
}

namespace hgcal {
  class HGCalFrameGenerator {
  public:
    explicit HGCalFrameGenerator(const edm::ParameterSet&);

    static edm::ParameterSetDescription description();

    void setRandomEngine(CLHEP::HepRandomEngine& rng);

    std::vector<uint32_t> produceECONEvent(uint32_t, const econd::ECONDInput&) const;
    const HGCalECONDEmulatorInfo& lastECONDEmulatedInfo() const { return last_econd_emul_info_; }

    /// Produce a S-link event from an input emulated event
    std::vector<uint64_t> produceSlinkEvent(uint32_t fed_id, const econd::ECONDInput&) const;
    const HGCalSlinkEmulatorInfo& lastSlinkEmulatedInfo() const { return last_slink_emul_info_; }

    const econd::EmulatorParameters& econdParams() const { return econd_; }

    struct SlinkParameters {
      std::vector<unsigned int> active_econds{};
      unsigned int boe_marker{0}, eoe_marker{0}, format_version{0};
    };
    const SlinkParameters& slinkParams() const { return slink_; }

  private:
    std::vector<bool> generateEnabledChannels() const;
    std::vector<uint32_t> generateERxData(const econd::ERxInput&) const;

    static constexpr size_t max_num_econds_ = 12;
    const bool passthrough_mode_;
    const bool expected_mode_;
    const bool characterisation_mode_;
    const bool matching_ebo_numbers_;
    const bool bo_truncated_;

    struct HeaderBits {
      bool bitO, bitB, bitE, bitT, bitH, bitS;
    };
    HeaderBits generateStatusBits() const;
    /// 8bit CRC for event header
    uint8_t computeCRC(const std::vector<uint32_t>&) const;

    econd::EmulatorParameters econd_;
    SlinkParameters slink_;
    CLHEP::HepRandomEngine* rng_{nullptr};  // NOT owning

    mutable HGCalECONDEmulatorInfo last_econd_emul_info_;
    mutable HGCalSlinkEmulatorInfo last_slink_emul_info_;
  };
}  // namespace hgcal

#endif
