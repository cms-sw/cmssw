#ifndef DataFormats_CSCDigi_CSCShowerDigi_h
#define DataFormats_CSCDigi_CSCShowerDigi_h

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <vector>

class CSCShowerDigi {
public:
  // Run-3 definitions as provided in DN-20-033
  enum Run3Shower { kInvalid = 0, kLoose = 1, kNominal = 2, kTight = 3 };
  // Shower types. and showers from OTMB/TMB are assigned with kLCTShower
  enum ShowerType {
    kInvalidShower = 0,
    kALCTShower = 1,
    kCLCTShower = 2,
    kLCTShower = 3,
    kEMTFShower = 4,
    kGMTShower = 5
  };

  /// Constructors
  CSCShowerDigi(const uint16_t inTimeBits,
                const uint16_t outTimeBits,
                const uint16_t cscID,
                const uint16_t bx = 0,
                const uint16_t showerType = 4,
                const uint16_t wireNHits = 0,
                const uint16_t compNHits = 0);
  /// default
  CSCShowerDigi();

  /// clear this Shower
  void clear();

  /// data
  bool isValid() const;

  bool isLooseInTime() const;
  bool isNominalInTime() const;
  bool isTightInTime() const;
  bool isLooseOutOfTime() const;
  bool isNominalOutOfTime() const;
  bool isTightOutOfTime() const;
  bool isValidShowerType() const;

  uint16_t bitsInTime() const { return bitsInTime_; }
  uint16_t bitsOutOfTime() const { return bitsOutOfTime_; }

  uint16_t getBX() const { return bx_; }
  uint16_t getCSCID() const { return cscID_; }
  uint16_t getShowerType() const { return showerType_; }
  uint16_t getWireNHits() const { return wireNHits_; }
  uint16_t getComparatorNHits() const { return comparatorNHits_; }

  /// set cscID
  void setCSCID(const uint16_t c) { cscID_ = c; }
  void setBX(const uint16_t bx) { bx_ = bx; }

private:
  uint16_t bitsInTime_;
  uint16_t bitsOutOfTime_;
  // 4-bit CSC chamber identifier
  uint16_t cscID_;
  uint16_t bx_;
  uint16_t showerType_;
  uint16_t wireNHits_;
  uint16_t comparatorNHits_;
};

std::ostream& operator<<(std::ostream& o, const CSCShowerDigi& digi);
#endif
