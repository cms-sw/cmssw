#ifndef DataFormats_FTLDigi_interface_ETLDigi_h
#define DataFormats_FTLDigi_interface_ETLDigi_h

#include <cstdint>
#include <ostream>

namespace etldigi {

  class ETLDigi {
  public:
    /**
     @short key to sort the collection
  */
    typedef uint32_t key_type;

    ETLDigi()
        : krawId_(0), kheader_(0), kstatus_(0), kcolID_(0), krowID_(0), kToAdata_(0), kToTdata_(0), kCALdata_(0) {}

    ETLDigi(uint32_t rawId,
            uint8_t header,
            uint8_t status,
            uint8_t colID,
            uint8_t rowID,
            uint16_t ToAdata,
            uint16_t ToTdata,
            uint16_t CALdata)
        : krawId_(rawId),
          kheader_(header),
          kstatus_(status),
          kcolID_(colID),
          krowID_(rowID),
          kToAdata_(ToAdata),
          kToTdata_(ToTdata),
          kCALdata_(CALdata) {}

    uint32_t krawId() const { return krawId_; }
    uint8_t kheader() const { return kheader_; }
    uint8_t kstatus() const { return kstatus_; }
    uint8_t kcolID() const { return kcolID_; }
    uint8_t krowID() const { return krowID_; }
    uint16_t kToAdata() const { return kToAdata_; }
    uint16_t kToTdata() const { return kToTdata_; }
    uint16_t kCALdata() const { return kCALdata_; }

    // needed for sorting in SortedCollection
    // For ETL, we modify the structure of raw DetID
    // to account for the pixel inside the LGAD
    // First 8 bits of DetId (common to all ETL modules)
    // are dropped, and row, column values are inserted
    // in the last 8 bits
    // Pixels are ordered per row then column
    uint32_t id() const {
      return (krawId_ << kGlobalOffset) + (((uint32_t)kcolID_ & kColMask) << kColOffset) +
             (((uint32_t)krowID_ & kRowMask) << kRowOffset);
    }

  private:
    uint32_t krawId_;  // Raw ID of the module/ETROC
    uint8_t kheader_;
    uint8_t kstatus_;  // status of the ETROC
    uint8_t kcolID_;   // LGAD pixel column
    uint8_t krowID_;   // LGAD pixel row
    uint16_t kToAdata_;
    uint16_t kToTdata_;
    uint16_t kCALdata_;

    static constexpr int kGlobalOffset = 8;
    static constexpr int kColMask = 0xF;
    static constexpr int kRowMask = 0xF;
    static constexpr int kColOffset = 0;
    static constexpr int kRowOffset = 4;
  };

  std::ostream& operator<<(std::ostream&, const ETLDigi&);

}  // namespace etldigi
#endif  // DataFormats_FTLDigi_interface_ETLDigi_h
