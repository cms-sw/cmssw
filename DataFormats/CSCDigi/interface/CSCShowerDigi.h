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

  /// Constructors
  CSCShowerDigi(const uint16_t inTimeBits, const uint16_t outTimeBits, const uint16_t cscID);
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

  uint16_t bitsInTime() const { return bitsInTime_; }
  uint16_t bitsOutOfTime() const { return bitsOutOfTime_; }

  uint16_t getCSCID() const { return cscID_; }

  /// set cscID
  void setCSCID(const uint16_t c) { cscID_ = c; }

private:
  uint16_t bitsInTime_;
  uint16_t bitsOutOfTime_;
  // 4-bit CSC chamber identifier
  uint16_t cscID_;
};

std::ostream& operator<<(std::ostream& o, const CSCShowerDigi& digi);
#endif
