#ifndef DTDigi_DTDigi_h
#define DTDigi_DTDigi_h

/** \class DTDigi
 *
 * Digi for Drift Tubes.
 * It can be initialized/set with a time in ns or a TDC count in 25/32 ns 
 * units.
 *  
 *  $Date: 2005/10/25 13:48:54 $
 *  $Revision: 1.2 $
 *
 * \author N. Amapane - INFN Torino
 *
 */

#include <boost/cstdint.hpp>

class DTDigi{

public:
  typedef uint32_t ChannelType;

  /// Lenght of packed fields
  enum packing{wire_s     = 7,
	       number_s   = 3,
	       counts_s   = 20,
	       trailer_s  = 2  // padding (unused)
  };

  /// The packed digi content  
  struct PackedDigiType {
    unsigned int wire     : wire_s;
    unsigned int number   : number_s;
    unsigned int counts   : counts_s;
    unsigned int trailer  : trailer_s; 
  };

  /// Construct from the wire#, the digi number and the TDC counts.
  /// number should identify uniquely multiple digis in the same cell.
  explicit DTDigi (int wire, int number, int nTDC);

  /// Construct from channel and counts.
  explicit DTDigi (ChannelType channel, int nTDC);

  /// Construct from a packed value
  DTDigi (PackedDigiType packed_value);

  /// Copy constructor
  DTDigi (const DTDigi& digi);

  /// Default construction.
  DTDigi ();

  /// Assignment operator
  DTDigi& operator=(const DTDigi& digi);

  /// Digis are equal if they are on the same cell and have same TDC count
  bool operator==(const DTDigi& digi) const;

  /// Get all digi content (incl. channel and digi number) in a packed fromat
  PackedDigiType packedData() const { return *(data()); }

  /// The channel identifier and the digi number packed together
  ChannelType channel() const ;

  /// R-Phi or R-Zed SuperLayer
  //  DTEnum::ViewCode viewCode() const ;

  /// Return wire number
  int wire() const;

  /// Identifies different digis within the same 
  int number() const;

  /// Get time in ns
  double time() const;

  /// Get raw TDC count
  int countsTDC() const;

  /// Set with a time in ns
  void setTime(double time);  

  /// Set with a TDC count
  void setCountsTDC (int nTDC);

  /// Setter for trailing bits at the end of the data words (currently unused)
  void setTrailer(int trailer);

  /// Print content of digi
  void print() const;

  /// Print the binary representation of the digi
  void dump() const;

private:
  friend class testDTDigis;

  // The value of one TDC count in ns
  static const double reso;

  // Set data words
  void set(int  wire, int number, int counts);

  // Set from a PackedDigiType
  void setData(PackedDigiType p);

  // Access to the packed data
  PackedDigiType* data();

  // Const access to the packed data
  const PackedDigiType* data() const;

  // Used to repack the channel number to an int
  struct ChannelPacking {
    unsigned int wire    : wire_s;
    unsigned int number  : number_s;
    unsigned int padding : trailer_s+counts_s;    // unused
  };

 public:
  // the packed data as seen by the persistency - should never be used 
  // directly, only by calling data()
// made public to be able to generate lcgdict, SA, 27/4/05
  struct PersistentPacking {
    uint32_t w1;
  };

 private:

  PersistentPacking persistentData;

};

#include<iostream>
// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const DTDigi& digi) {
  return o << " " << digi.wire()
	   << " " << digi.time()
	   << " " << digi.number();
}
#endif

