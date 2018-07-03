#ifndef CSCWireDigi_CSCWireDigi_h
#define CSCWireDigi_CSCWireDigi_h

/**\class CSCWireDigi
 *
 * Digi for CSC anode wires. 
 *
 */

#include <vector>
#include <iosfwd>
#include <cstdint>

class CSCWireDigi{

public:

  /// Constructors
  
  CSCWireDigi (int wire, unsigned int tbinb);  /// wiregroup#, tbin bit word
  CSCWireDigi ();                     /// default

  /// return wiregroup number. counts from 1.
  int getWireGroup() const {return wire_;}
  /// return BX assigned for the wire group (16 upper bits from the wire group number)
  int getWireGroupBX() const {return wireBX_;}
  /// return BX-wiregroup number combined 
  /// (16 upper bits - BX + 16 lower bits - wire group number)
  int getBXandWireGroup() const {return wireBXandWires_;}
  /// return the word with time bins bits
  unsigned int getTimeBinWord() const {return tbinb_;}
  /// return tbin number, (obsolete, use getTimeBin() instead)
  int getBeamCrossingTag() const;
  /// return first tbin ON number
  int getTimeBin()         const;
  /// return vector of time bins ON
  std::vector<int> getTimeBinsOn() const;

  /// Print content of digi
  void print() const;

  /// set wiregroup number
  void setWireGroup(unsigned int wiregroup) {wire_= wiregroup;}


private:

  int wire_;
  uint32_t tbinb_;
  /// BX in the wire digis (16 upper bits from the wire group number)
  int wireBXandWires_;
  int wireBX_;

};

std::ostream & operator<<(std::ostream & o, const CSCWireDigi& digi);

#endif
