#ifndef CSCDMBStatusDigi_CSCDMBStatusDigi_h
#define CSCDMBStatusDigi_CSCDMBStatusDigi_h

/** \class CSCDMBStatusDigi
 *
 *  Digi for CSC DMB info available in DDU
 *  
 *
 */

#include <vector>
#include <iosfwd>
#include <cstdint>

class CSCDMBStatusDigi {
public:
  /// Constructor for all variables
  CSCDMBStatusDigi(const uint16_t* header, const uint16_t* trailer);

  /// Default constructor.
  CSCDMBStatusDigi() {}

  /// Data Accessors
  const uint16_t* header() const { return header_; }
  const uint16_t* trailer() const { return trailer_; }

private:
  uint16_t header_[8];
  uint16_t trailer_[8];
};

std::ostream& operator<<(std::ostream& o, const CSCDMBStatusDigi& digi);

#endif
