#ifndef DataFormats_FEDRawData_SLinkRocketHeaders_h
#define DataFormats_FEDRawData_SLinkRocketHeaders_h

#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <span>

/*
 * DTH Orbit header, event fragment trailer and SlinkRocket Header and Trailer accompanying
 * slink payload. Format that is sent is is low-endian for multi-byte fields
 *
 * Version 1 DTH and Version 3 SLinkRocket
 *
 * */

constexpr uint32_t SLR_MAX_EVENT_LEN = (1 << 20) - 1;
constexpr uint32_t SLR_WORD_NUM_BYTES_SHIFT = 4;

//SLinkExpress classes

//begin and end event markers
constexpr uint8_t SLR_BOE = 0x55;
constexpr uint8_t SLR_EOE = 0xaa;

//minimal SLinkRocket version overlay
class SLinkRocketHeader_version {
public:
  SLinkRocketHeader_version() {}
  uint8_t version() const { return version_; }
  bool verifyMarker() const { return boe_ == SLR_BOE; }

private:
  uint32_t source_id_;
  uint16_t l1a_types_;
  uint8_t phys_type_;
  uint8_t emu_status_ : 2, res1_ : 6;
  uint64_t event_id_ : 44, res2_ : 8, version_ : 4, boe_ : 8;
};

class SLinkRocketHeader_v3 {
public:
  SLinkRocketHeader_v3(uint32_t source_id, uint16_t l1a_types, uint8_t l1a_phys, uint8_t emu_status, uint64_t event_id)
      : source_id_(source_id),
        l1a_types_(l1a_types),
        phys_type_(l1a_phys),
        emu_status_(emu_status),
        event_id_(event_id) {}

  uint32_t sourceID() const { return source_id_; }
  uint16_t l1aTypes() const { return l1a_types_; }
  uint8_t l1aPhysType() const { return phys_type_; }
  uint8_t emuStatus() const { return emu_status_; }
  uint64_t globalEventID() const { return event_id_; }
  uint8_t version() const { return version_; }
  bool verifyMarker() const { return boe_ == SLR_BOE; }

private:
  uint32_t source_id_;
  uint16_t l1a_types_;
  uint8_t phys_type_;
  uint8_t emu_status_ : 2, res1_ : 6;
  uint64_t event_id_ : 44, res2_ : 8, version_ : 4 = 3, boe_ : 8 = SLR_BOE;
};

class SLinkRocketTrailer_v3 {
public:
  SLinkRocketTrailer_v3(
      uint16_t status, uint16_t crc, uint32_t orbit_id, uint16_t bx_id, uint32_t evtlen_word_count, uint16_t daq_crc)
      : crc_(crc), orbit_id_(orbit_id), bx_id_(bx_id), event_length_wcount_(evtlen_word_count), daq_crc_(daq_crc) {
    status_.all_ = status;
  }

  uint16_t status() const { return status_.all_; }
  uint16_t crc() const { return crc_; }
  uint32_t orbitID() const { return orbit_id_; }
  uint16_t bxID() const { return bx_id_; }
  uint32_t eventLenBytes() const { return uint32_t(event_length_wcount_) << SLR_WORD_NUM_BYTES_SHIFT; }
  uint16_t daqCRC() const { return daq_crc_; }
  bool verifyMarker() const { return eoe_ == SLR_EOE; }

private:
  union {
    struct {
      uint16_t fed_crc_error_ : 1, /* FED CRC error was detected by DTH and corrected */
          slink_crc_error_
          : 1, /* Set when the slink receviver finds a mistmatch between calculated crc and daq_crc. It should never happen */
          source_id_error_ : 1, /* SOURCE_ID is not expected */
          sync_lost_error_ : 1, /* Sync lost detected by DTH */
          fragment_cut_ : 1,    /* Fragment was cut */
          res_ : 11;
    } bits_;
    uint16_t all_;
  } status_;
  uint16_t crc_; /* CRC filled by the FED */
  uint32_t orbit_id_;
  uint32_t bx_id_ : 12, event_length_wcount_
      : 20; /* Length is encoded in multiples of 128 bits (16 bytes). I.e needs to be shifter by 4 */
  uint32_t reserved_ : 8, daq_crc_ : 16, /* CRC filled by the slink sender */
      eoe_ : 8 = SLR_EOE;
};

/*
 * version-independent header view parent class
 * */

class SLinkRocketHeaderView {
public:
  virtual ~SLinkRocketHeaderView() = default;
  virtual uint32_t sourceID() const = 0;
  virtual uint16_t l1aTypes() const = 0;
  virtual uint8_t l1aPhysType() const = 0;
  virtual uint8_t emuStatus() const = 0;
  virtual uint64_t globalEventID() const = 0;
  virtual uint8_t version() const = 0;
  virtual bool verifyMarker() const = 0;
};

/*
 * header v3 view abstraction
 * */

class SLinkRocketHeaderView_v3 : public SLinkRocketHeaderView {
public:
  SLinkRocketHeaderView_v3(const void* header) : header_(static_cast<const SLinkRocketHeader_v3*>(header)) {}

  uint32_t sourceID() const override { return header_->sourceID(); }
  uint16_t l1aTypes() const override { return header_->l1aTypes(); }
  uint8_t l1aPhysType() const override { return header_->l1aPhysType(); }
  uint8_t emuStatus() const override { return header_->emuStatus(); }
  uint64_t globalEventID() const override { return header_->globalEventID(); }
  uint8_t version() const override { return header_->version(); }
  bool verifyMarker() const override { return header_->verifyMarker(); }

private:
  const SLinkRocketHeader_v3* header_;
};

static inline std::unique_ptr<SLinkRocketHeaderView> makeSLinkRocketHeaderView(const void* buf) {
  auto version = static_cast<const SLinkRocketHeader_version*>(buf)->version();
  if (version == 3)
    return std::unique_ptr<SLinkRocketHeaderView>(
        static_cast<SLinkRocketHeaderView*>(new SLinkRocketHeaderView_v3(buf)));
  throw cms::Exception("SLinkRocketHeaderView::makeView")
      << "unknown SLinkRocketHeader version: " << (unsigned int)version;
}

static inline std::unique_ptr<SLinkRocketHeaderView> makeSLinkRocketHeaderView(std::span<const unsigned char> const& s) {
  if (s.size() < sizeof(SLinkRocketHeader_version))
    throw cms::Exception("SLinkRocketHeaderView::makeView")
        << "size is smaller than SLink header version fields: " << s.size();
  auto version = static_cast<const SLinkRocketHeader_version*>(static_cast<const void*>(&s[0]))->version();
  if (version == 3 && s.size() != sizeof(SLinkRocketHeader_v3))
    throw cms::Exception("SLinkRocketHeaderView::makeView") << "SLinkRocketHeader v3 size mismatch: got " << s.size()
                                                            << " expected:" << sizeof(SLinkRocketHeader_v3) << " bytes";

  return makeSLinkRocketHeaderView(static_cast<const void*>(&s[0]));
}

/*
 * version-independent trailer view parent class
 * */

class SLinkRocketTrailerView {
public:
  virtual ~SLinkRocketTrailerView() = default;
  virtual uint16_t status() const = 0;
  virtual uint16_t crc() const = 0;
  virtual uint32_t orbitID() const = 0;
  virtual uint16_t bxID() const = 0;
  virtual uint32_t eventLenBytes() const = 0;
  virtual uint16_t daqCRC() const = 0;
  virtual bool verifyMarker() const = 0;
};

/*
 * trailer v3 view abstraction
 * */

class SLinkRocketTrailerView_v3 : public SLinkRocketTrailerView {
public:
  SLinkRocketTrailerView_v3(const void* trailer) : trailer_(static_cast<const SLinkRocketTrailer_v3*>(trailer)) {}
  uint16_t status() const override { return trailer_->status(); }
  uint16_t crc() const override { return trailer_->crc(); }
  uint32_t orbitID() const override { return trailer_->orbitID(); }
  uint16_t bxID() const override { return trailer_->bxID(); }
  uint32_t eventLenBytes() const override { return trailer_->eventLenBytes(); }
  uint16_t daqCRC() const override { return trailer_->daqCRC(); }
  bool verifyMarker() const override { return trailer_->verifyMarker(); }

private:
  const SLinkRocketTrailer_v3* trailer_;
};

static inline std::unique_ptr<SLinkRocketTrailerView> makeSLinkRocketTrailerView(const void* buf, uint8_t version) {
  if (version == 3)
    return std::unique_ptr<SLinkRocketTrailerView>(
        static_cast<SLinkRocketTrailerView*>(new SLinkRocketTrailerView_v3(buf)));
  throw cms::Exception("SLinkRocketTrailerView::makeView")
      << "unknown SLinkRocketHeader version: " << (unsigned int)version;
}

static inline std::unique_ptr<SLinkRocketTrailerView> makeSLinkRocketTrailerView(
    std::span<const unsigned char> const& s, uint8_t version) {
  if (version == 3 && s.size() < sizeof(SLinkRocketTrailer_v3))
    throw cms::Exception("SLinkRocketTrailerView::makeView")
        << "SLinkRocketTrailer v3 size mismatch: got " << s.size() << " expected " << sizeof(SLinkRocketTrailer_v3)
        << " bytes";
  return makeSLinkRocketTrailerView(static_cast<const void*>(&s[0]), version);
}
#endif
