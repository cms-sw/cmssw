#ifndef EventFilter_HGCalRawToDigi_HGCalRawDataPackingTools_h
#define EventFilter_HGCalRawToDigi_HGCalRawDataPackingTools_h

#include "EventFilter/HGCalRawToDigi/interface/SlinkTypes.h"

namespace hgcal {
  namespace econd {
    struct ERxData;
    /// pack the ROC data to the ECON-D format dependending on:
    ///   - characterization mode : (TcTp + ADC + TOT + TOA) fixed 32b
    ///   - normal mode : size and fields depend on the TcTp flags
    /// \note based on Fig. 20 of ECON-D specifications
    /// \return a vector of new words
    std::vector<uint32_t> produceERxData(
        const ERxChannelEnable&, const ERxData&, bool passZS, bool passZSm1, bool hasToA, bool char_mode);

    /// returns the words for a new eRx header
    /// \note based on Fig. 33 of ECON-D specifications
    /// \return a vector with 1 or 2 32b words
    std::vector<uint32_t> eRxSubPacketHeader(uint8_t stat,
                                             uint8_t ham,
                                             bool bitE,
                                             uint16_t common_mode0,
                                             uint16_t common_mode1,
                                             const ERxChannelEnable& channel_enable);
    std::vector<uint32_t> eRxSubPacketHeader(
        uint8_t stat, uint8_t ham, bool bitE, uint16_t common_mode0, uint16_t common_mode1, uint64_t channels_map);

    /// builds the two ECON-D header words
    /// \note based on Fig. 33 of the ECON-D specs
    /// \return a vector of size 2 with the ECON-D header
    std::vector<uint32_t> eventPacketHeader(uint16_t header,
                                            uint16_t payload,
                                            bool bitP,
                                            bool bitE,
                                            uint8_t ht,
                                            uint8_t ebo,
                                            bool bitM,
                                            bool bitT,
                                            uint8_t hamming,
                                            uint16_t bx,
                                            uint16_t l1a,
                                            uint8_t orb,
                                            bool bitS,
                                            uint8_t RR);
    /// builds a trailing idle word
    /// \note based on Fig. 33 of the ECON-D specs
    /// \return a 32b word with the idle word
    uint32_t buildIdleWord(uint8_t bufStat, uint8_t err, uint8_t rr, uint32_t progPattern);
  }  // namespace econd

  namespace backend {
    enum ECONDPacketStatus {
      Normal = 0x0,
      PayloadCRCError = 0x1,
      EventIDMismatch = 0x2,
      EBTimeout = 0x4,
      BCIDOrbitIDMismatch = 0x5,
      MainBufferOverflow = 0x6,
      InactiveECOND = 0x7
    };

    /// builds the capture block header (see page 16 of "HGCAL BE DAQ firmware description")
    /// \return a vector of size 2 with the 2 32b words of the capture block header
    std::vector<uint32_t> buildCaptureBlockHeader(uint32_t bunch_crossing,
                                                  uint32_t event_counter,
                                                  uint32_t orbit_counter,
                                                  const std::vector<ECONDPacketStatus>& econd_statuses);

    /// builds the slink frame header (128 bits header = 4 words)
    /// \return a vector with 4 32b words
    std::vector<uint32_t> buildSlinkHeader(
        uint8_t boe, uint8_t v, uint64_t global_event_id, uint32_t content_id, uint32_t fed_id);

    /// builds the slink frame trailer (128 bits trailer = 4 words)
    /// \return a vector with 4 32b words
    std::vector<uint32_t> buildSlinkTrailer(uint8_t eoe,
                                            uint16_t daqcrc,
                                            uint32_t event_length,
                                            uint16_t bxid,
                                            uint32_t orbit_id,
                                            uint16_t crc,
                                            uint16_t status);

    enum SlinkEmulationFlag { Subsystem = 0, SlinkRocketSenderCore = 1, DTH = 2 };
    /// builds the slink rocket event data content ID
    /// \return a 32b word
    uint32_t buildSlinkContentId(SlinkEmulationFlag, uint8_t l1a_subtype, uint16_t l1a_fragment_cnt);

    /// builds the SlinkRocket sender core status field
    /// \return a 16b word
    uint16_t buildSlinkRocketStatus(
        bool fed_crc_err, bool slinkrocket_crc_err, bool source_id_err, bool sync_lost, bool fragment_trunc);
  }  // namespace backend
}  // namespace hgcal

#endif
