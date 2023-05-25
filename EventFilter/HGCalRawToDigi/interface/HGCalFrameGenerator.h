/****************************************************************************
 *
 * This is a part of HGCAL offline software.
 * Authors:
 *   Laurent Forthomme, CERN
 *
 ****************************************************************************/

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
  namespace econd {
    class Emulator;
  }
  /// A S-link/ECON-D payload generator helper
  class HGCalFrameGenerator {
  public:
    explicit HGCalFrameGenerator(const edm::ParameterSet&);

    static edm::ParameterSetDescription description();

    /// Set the random number generator engine
    void setRandomEngine(CLHEP::HepRandomEngine& rng);
    /// Set the emulation source for ECON-D frames
    void setEmulator(econd::Emulator&);

    /// Produce a S-link event from an emulated event
    std::vector<uint64_t> produceSlinkEvent(unsigned int fed_id) const;
    /// Produce a capture block from an emulated event
    /// \params[in] cb_id capture block identifier
    std::vector<uint64_t> produceCaptureBlockEvent(unsigned int cb_id) const;
    /// Produce a ECON-D event from an emulated event
    /// \param[in] econd_id ECON-D identifier
    /// \param[in] cb_id capture block identifier
    std::vector<uint64_t> produceECONEvent(unsigned int econd_id, unsigned int cb_id = 0) const;

    /// Retrieve the last ECON-D event emulated
    const econd::ECONDInput& lastECONDEmulatedInput() const { return last_emul_event_; }
    /// Retrieve the metadata generated along with the last S-link emulated payload
    const HGCalSlinkEmulatorInfo& lastSlinkEmulatedInfo() const { return last_slink_emul_info_; }

    /// List of S-link operational parameters for emulation
    struct SlinkParameters {
      std::vector<unsigned int> active_econds{};
      unsigned int boe_marker{0}, eoe_marker{0}, format_version{0}, num_capture_blocks{1};
      bool store_header_trailer{true};
    };
    /// List of S-link operational parameters for emulation
    const SlinkParameters& slinkParams() const { return slink_params_; }
    /// List of ECON-D operational parameters for emulation
    const std::map<unsigned int, econd::EmulatorParameters>& econdParams() const { return econd_params_; }

  private:
    econd::ERxChannelEnable generateEnabledChannels(unsigned int) const;
    std::vector<uint32_t> generateERxData(unsigned int,
                                          const econd::ERxInput&,
                                          std::vector<econd::ERxChannelEnable>&) const;

    static constexpr size_t kMaxNumECONDs = 12;

    struct HeaderBits {
      bool bitO, bitB, bitE, bitT, bitH, bitS;
    };
    HeaderBits generateStatusBits(unsigned int) const;
    /// 32bit CRC
    uint32_t computeCRC(const std::vector<uint32_t>&) const;

    SlinkParameters slink_params_;
    std::map<unsigned int, econd::EmulatorParameters> econd_params_;

    CLHEP::HepRandomEngine* rng_{nullptr};    // NOT owning
    mutable econd::Emulator* emul_{nullptr};  // NOT owning

    mutable HGCalSlinkEmulatorInfo last_slink_emul_info_;
    mutable econd::ECONDInput last_emul_event_;
  };
}  // namespace hgcal

#endif
