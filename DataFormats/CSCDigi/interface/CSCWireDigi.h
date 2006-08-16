#ifndef CSCWireDigi_CSCWireDigi_h
#define CSCWireDigi_CSCWireDigi_h

/**\class CSCWireDigi
 *
 * Digi for CSC anode wires. 
 *
 * \author N. Terentiev, CMU
 */


#include <boost/cstdint.hpp>

class CSCWireDigi{

public:

  /// Constructors
  
  CSCWireDigi (int wire, int tbin);  /// from the wiregroup#, tbin#
  CSCWireDigi ();                             /// default


  /// return wiregroup number
  int getWireGroup() const {return wire_;}
  /// return tbin number
  int getBeamCrossingTag() const {return tbin_;}
  /// return tbin number, consider getBeamCrossingTag() obsolete
  int getTimeBin()         const {return tbin_;}

  /// Print content of digi
  void print() const;

  /// set wiregroup number
  void setWireGroup(unsigned int wiregroup) {wire_= wiregroup;}


private:
  friend class testCSCDigis; //@@ Do we really want friend access for a test?
  uint16_t wire_;
  uint16_t tbin_;

};

#include<iostream>

inline std::ostream & operator<<(std::ostream & o, const CSCWireDigi& digi) {
  return o << " CSC Wire " << digi.getWireGroup()
	   << " CSC Wire Time Bin " << digi.getTimeBin();
}
#endif
