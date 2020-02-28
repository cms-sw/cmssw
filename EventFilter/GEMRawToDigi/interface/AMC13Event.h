#ifndef EventFilter_GEMRawToDigi_AMC13Event_h
#define EventFilter_GEMRawToDigi_AMC13Event_h
#include <vector>
#include "AMCdata.h"

namespace gem {

  union CDFHeader {
    uint64_t word;
    struct {
      uint64_t fov : 8;        // not used
      uint64_t sourceId : 12;  // FED number assigned by CDAQ
      uint64_t bxId : 12;      // BX number, Reset by BC0
      uint64_t lv1Id : 24;     // L1A / event number, Reset by EC0
      uint64_t eventType : 4;  // Event Type (1 for normal, 2 for calibration)
      uint64_t cb5 : 4;        // 0x5
    };
  };
  union AMC13Header {
    uint64_t word;
    struct {
      uint64_t cb0 : 4;         // 0x0
      uint64_t orbitN : 32;     // Orbit Number
      uint64_t reserved0 : 16;  // reserved
      uint64_t nAMC : 4;        // Number of AMCs following (0 to 12)
      uint64_t calType : 4;     // Calibration event type
      uint64_t uFov : 4;        // Format version: 0x1
    };
  };
  union AMC13Trailer {
    uint64_t word;
    struct {
      uint64_t bxIdT : 12;  // bx id
      uint64_t lv1IdT : 8;  // level 1 id
      uint64_t blkN : 8;    // block number
      uint64_t crc32 : 36;  // Overall CRC (first 32 bits)
    };
  };
  union CDFTrailer {
    uint64_t word;
    struct {
      uint64_t tts : 8;         // tts (first 4 bits)
      uint64_t evtStat : 4;     // event status
      uint64_t crcCDF : 20;     // CDF crc (first 16 bits)
      uint64_t evtLength : 24;  // event length
      uint64_t eventType : 4;   // Event Type
      uint64_t cbA : 4;         // 0xA
    };
  };

  class AMC13Event {
  public:
    AMC13Event() : cdfh_(0), amc13h_(0), amc13t_(0), cdft_(0) {}
    ~AMC13Event() {
      amcHeaders_.clear();
      amcs_.clear();
    }

    void setCDFHeader(uint64_t word) { cdfh_ = word; }
    void setCDFHeader(uint8_t Evt_ty, uint32_t LV1_id, uint16_t BX_id, uint16_t Source_id);
    uint64_t getCDFHeader() const { return cdfh_; }

    void setAMC13Header(uint64_t word) { amc13h_ = word; }
    void setAMC13Header(uint8_t CalTyp, uint8_t nAMC, uint32_t OrN);
    uint64_t getAMC13Header() const { return amc13h_; }

    void setAMC13Trailer(uint64_t word) { amc13t_ = word; }
    void setAMC13Trailer(uint8_t Blk_NoT, uint8_t LV1_idT, uint16_t BX_idT);
    uint64_t getAMC13Trailer() const { return amc13t_; }

    void setCDFTrailer(uint64_t word) { cdft_ = word; }
    void setCDFTrailer(uint32_t EvtLength);
    uint64_t getCDFTrailer() const { return cdft_; }

    uint16_t bxId() const { return CDFHeader{cdfh_}.bxId; }
    uint32_t lv1Id() const { return CDFHeader{cdfh_}.lv1Id; }
    uint16_t sourceId() const { return CDFHeader{cdfh_}.sourceId; }

    uint8_t nAMC() const { return AMC13Header{amc13h_}.nAMC; }

    const std::vector<uint64_t>* getAMCheaders() const { return &amcHeaders_; }
    void addAMCheader(uint64_t word);
    void addAMCheader(uint32_t AMC_size, uint8_t Blk_No, uint8_t AMC_No, uint16_t BoardID);

    const std::vector<AMCdata>* getAMCpayloads() const { return &amcs_; }
    void addAMCpayload(const AMCdata& a) { amcs_.push_back(a); }

  private:
    uint64_t cdfh_;    // CDFHeader
    uint64_t amc13h_;  // AMC13Header
    uint64_t amc13t_;  // AMC13Trailer
    uint64_t cdft_;    // CDFTrailer

    // AMC headers
    std::vector<uint64_t> amcHeaders_;
    // AMCs payload
    std::vector<AMCdata> amcs_;
  };
}  // namespace gem
#endif
