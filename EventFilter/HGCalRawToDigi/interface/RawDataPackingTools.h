#ifndef EventFilter_HGCalRawToDigi_RawDataPackingTools_h
#define EventFilter_HGCalRawToDigi_RawDataPackingTools_h

#include <cstdint>
#include <vector>

namespace hgcal {
  namespace econd {
    /**
     * \short packs the ROC data to the ECON-D format dependending on
     *   - characterization mode : (TcTp + ADC + TOT + TOA) fixed 32b
     *   - normal mode : size and fields depend on the TcTb flags
     * \note based on Fig. 20 of ECON-D specifications
     * \return a vector of new words (up to 2 in case one needs to use the next 32b
     *   the msb is updated as the reference is passed
     */
    std::vector<uint32_t> addChannelData(uint8_t &msb,
                                         uint16_t tctp,
                                         uint16_t adc,
                                         uint16_t tot,
                                         uint16_t adcm,
                                         uint16_t toa,
                                         bool passZS,
                                         bool passZSm1,
                                         bool hasToA,
                                         bool charmode);

    /**
     * \short returns the words for a new eRx header
     * \note based on Fig. 33 of ECON-D specifications
     * \return a vector with 1 or 2 32b words
     */
    std::vector<uint32_t> eRxSubPacketHeader(
        uint16_t stat, uint16_t ham, bool bitE, uint16_t cm0, uint16_t cm1, std::vector<bool> chmap);

    /**
     * \short builds the two ECON-D header words
     * \note based on Fig. 33 of the ECON-D specs
     * \return a vector of size 2 with the ECON-D header
     */
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
                                            uint8_t RR,
                                            uint8_t ehCRC);
    /**
     * \short builds a trailing idle word
     * \note based on Fig. 33 of the ECON-D specs
     * \return a 32b word with the idle word
     */
    uint32_t buildIdleWord(uint8_t bufStat, uint8_t err, uint8_t rr, uint32_t progPattern);
  }  // namespace econd

  namespace backend {
    /**
     * \short builds the capture block header (see page 16 of "HGCAL BE DAQ firmware description")
     * \return a vector of size 2 with the 2 32b words of the capture block header
     */
    std::vector<uint32_t> buildCaptureBlockHeader(uint32_t bc,
                                                  uint32_t ec,
                                                  uint32_t oc,
                                                  std::vector<uint8_t> &econdStatus);

    /**
     * \short builds the slink frame header (128 bits header = 4 words)
     * \return a vector with 4 32b words
     */
    std::vector<uint32_t> buildSlinkHeader(
        uint8_t boe, uint8_t v, uint8_t r8, uint64_t global_event_id, uint8_t r6, uint32_t content_id, uint32_t fed_id);

    /**
     * \short builds the slink frame trailer (128 bits trailer = 4 words)
     * \return a vector with 4 32b words
     */
    std::vector<uint32_t> buildSlinkTrailer(uint8_t eoe,
                                            uint8_t daqcrc,
                                            uint8_t trailer_r,
                                            uint64_t event_length,
                                            uint8_t bxid,
                                            uint32_t orbit_id,
                                            uint32_t crc,
                                            uint32_t status);
  }  // namespace backend
}  // namespace hgcal

#endif
