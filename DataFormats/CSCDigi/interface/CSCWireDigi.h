#ifndef CSCWireDigi_CSCWireDigi_h
#define CSCWireDigi_CSCWireDigi_h

/**\class CSCWireDigi
 *
 * Digi for CSC anode wires. 
 * Based on modified DTDigi.
 *
 * $Date: 2006/04/05 22:17:19 $
 * $Revision: 1.3 $
 *
 * \author N. Terentiev, CMU
 */


#include <boost/cstdint.hpp>

class CSCWireDigi{

public:

  /// Constructors
  
  explicit CSCWireDigi (int wire, int tbin);  /// from the wire#, tbin#
  CSCWireDigi ();                             /// default


  /// return wire number
  int getWireGroup() const {return wire_;}
  /// return tbin number
  int getBeamCrossingTag() const {return tbin_;}
  /// return tbin number, consider getBeamCrossingTag() obsolete
  int getTimeBin()         const {return tbin_;}

  /// Print content of digi
  void print() const;

private:
  friend class testCSCDigis;
  uint16_t wire_;
  uint16_t tbin_;

};

#include<iostream>
  /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCWireDigi& digi) {
  return o << " CSC Wire " << digi.getWireGroup()
	   << " CSC Wire Time Bin " << digi.getTimeBin();
}
#endif
