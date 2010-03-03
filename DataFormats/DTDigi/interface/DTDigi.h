#ifndef DTDigi_DTDigi_h
#define DTDigi_DTDigi_h

/** \class DTDigi
 *
 * Digi for Drift Tubes.
 * It can be initialized/set with a time in ns or a TDC count in 25/32 ns 
 * units.
 *  
 *  $Date: 2006/04/05 15:03:07 $
 *  $Revision: 1.7 $
 *
 * \author N. Amapane, G. Cerminara, M. Pelliccioni - INFN Torino
 *
 */

#include <boost/cstdint.hpp>

class DTDigi{

public:
  typedef uint32_t ChannelType;
  
  /// Construct from the wire#, the TDC counts and the digi number.
  /// number should identify uniquely multiple digis in the same cell.
  explicit DTDigi (int wire, int nTDC, int number=0);

  /// Construct from the wire#, the time (ns) and the digi number.
  /// time is converted in TDC counts (1 TDC = 25./32. ns)
  /// number should identify uniquely multiple digis in the same cell.
  explicit DTDigi (int wire, double tdrift, int number=0);

  /// Construct from channel and counts.
  explicit DTDigi (ChannelType channel, int nTDC);

  /// Default construction.
  DTDigi ();

  /// Digis are equal if they are on the same cell and have same TDC count
  bool operator==(const DTDigi& digi) const;

  /// The channel identifier and the digi number packed together
  ChannelType channel() const ;

  /// Return wire number
  int wire() const;

  /// Identifies different digis within the same 
  int number() const;

  /// Get time in ns
  double time() const;

  /// Get raw TDC count
  uint32_t countsTDC() const;

  /// Set with a time in ns
  void setTime(double time);  

  /// Set with a TDC count
  void setCountsTDC (int nTDC);

  /// Print content of digi
  void print() const;

private:
  friend class testDTDigis;

  // The value of one TDC count in ns
  static const double reso;

  // Masks&shifts for the channel identifier
  static const uint32_t wire_mask   = 0xffff0000;
  static const uint32_t number_mask = 0xffff;
  static const uint32_t wire_offset = 16;

  uint16_t theWire;   // channel number
  uint32_t theCounts; // TDC count, up to 20 bits actually used
  uint16_t theNumber; // counter for digis in the same cell
};

#include<iostream>
inline std::ostream & operator<<(std::ostream & o, const DTDigi& digi) {
  return o << " " << digi.wire()
	   << " " << digi.time()
	   << " " << digi.number();
}
#endif

