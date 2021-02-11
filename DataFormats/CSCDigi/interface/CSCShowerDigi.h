#ifndef CSCDigi_CSCShowerDigi_h
#define CSCDigi_CSCShowerDigi_h

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <vector>

class CSCShowerDigi {
public:
  // Run-3 definitions as provided in DN-20-033
  enum Run3Shower { kInvalid = 0, kLoose = 1, kNominal = 2, kTight = 3 };

  /// Constructors
  CSCShowerDigi(const uint8_t bits, const uint16_t cscID);

  /// default
  CSCShowerDigi();

  // any loose shower is valid
  bool isValid() const;

  /// data
  uint16_t bits() const { return bits_; }
  bool isLoose() const;
  bool isNominal() const;
  bool isTight() const;

  // trigger chamber
  uint16_t getCSCID() const { return cscID_; }

  /// set trigger chamber
  void setCSCID(const uint16_t c) { cscID_ = c; }

private:
  // 4 bits hit counter
  uint8_t bits_;
  // 4-bit CSC chamber identifier
  uint8_t cscID_;
};

std::ostream& operator<<(std::ostream& o, const CSCShowerDigi& digi);
#endif
