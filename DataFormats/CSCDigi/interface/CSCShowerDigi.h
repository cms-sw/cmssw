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
  enum BitMask { kInTimeMask = 0x2, kOutTimeMask = 0x2 };
  enum BitShift { kInTimeShift = 0, kOutTimeShift = 2 };

  /// Constructors
  CSCShowerDigi(const uint16_t inTimeBits, const uint16_t outTimeBits, const uint16_t cscID);
  /// default
  CSCShowerDigi();

  /// clear this Shower
  void clear() { bits_ = 0; }

  /// data
  bool isValid() const;

  bool isLooseInTime() const;
  bool isNominalInTime() const;
  bool isTightInTime() const;
  bool isLooseOutTime() const;
  bool isNominalOutTime() const;
  bool isTightOutTime() const;

  uint16_t bits() const { return bits_; }
  uint16_t bitsInTime() const;
  uint16_t bitsOutTime() const;

  uint16_t getCSCID() const { return cscID_; }

  /// set cscID
  void setCSCID(const uint16_t c) { cscID_ = c; }

private:
  void setDataWord(const uint16_t newWord, uint16_t& word, const unsigned shift, const unsigned mask);
  uint16_t getDataWord(const uint16_t word, const unsigned shift, const unsigned mask) const;

  uint16_t bits_;
  // 4-bit CSC chamber identifier
  uint16_t cscID_;
};

std::ostream& operator<<(std::ostream& o, const CSCShowerDigi& digi);
#endif
