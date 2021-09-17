#ifndef DTDigi_DTDigi_h
#define DTDigi_DTDigi_h

/** \class DTDigi
 *
 * Digi for Drift Tubes.
 * It can be initialized/set with a time in ns or a TDC count in
 * the specified base (ie number of counts/BX).
 *  
 *
 * \author N. Amapane, G. Cerminara, M. Pelliccioni - INFN Torino
 *
 */

#include <cstdint>

class DTDigi {
public:
  //  typedef uint32_t ChannelType;

  /// Construct from the wire#, the TDC counts and the digi number.
  /// number should identify uniquely multiple digis in the same cell.
  explicit DTDigi(int wire, int nTDC, int number = 0, int base = 32);

  // Construct from the wire#, the time (ns) and the digi number.
  // time is converted in TDC counts (1 TDC = 25./base ns)
  // number should identify uniquely multiple digis in the same cell.
  explicit DTDigi(int wire, double tdrift, int number = 0, int base = 32);

  // Construct from channel and counts.
  //  explicit DTDigi (ChannelType channel, int nTDC);

  /// Default construction.
  DTDigi();

  /// Digis are equal if they are on the same cell and have same TDC count
  bool operator==(const DTDigi& digi) const;

  // The channel identifier and the digi number packed together
  //  ChannelType channel() const ;

  /// Return wire number
  int wire() const;

  /// Identifies different digis within the same cell
  int number() const;

  /// Get time in ns
  double time() const;

  /// Get raw TDC count
  int32_t countsTDC() const;

  /// Get the TDC unit value in ns
  double tdcUnit() const;

  /// Get the TDC base (counts per BX)
  int tdcBase() const;

  /// Print content of digi
  void print() const;

private:
  friend class testDTDigis;

  // The value of one TDC count in ns
  static const double reso;

  int32_t theCounts;   // TDC count, in units given by 1/theTDCBase
  uint16_t theWire;    // channel number
  uint8_t theNumber;   // counter for digis in the same cell
  uint8_t theTDCBase;  // TDC base (counts per BX; 32 in Ph1 or 30 in Ph2)
};

#include <iostream>
#include <cstdint>
inline std::ostream& operator<<(std::ostream& o, const DTDigi& digi) {
  return o << " " << digi.wire() << " " << digi.time() << " " << digi.number();
}
#endif
