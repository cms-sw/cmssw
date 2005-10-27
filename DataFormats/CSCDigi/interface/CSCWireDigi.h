#ifndef CSCWireDigi_CSCWireDigi_h
#define CSCWireDigi_CSCWireDigi_h

/**\class CSCWireDigi
 *
 * Digi for CSC anode wires. 
 * Based on modified DTDigi.
 *
 * $Date$
 * $Revision$
 *
 * \author N. Terentiev, CMU
 */

class CSCWireDigi{

public:

  typedef unsigned int ChannelType;

  /// Enum, structures

      /// Length of packed fields
  enum packing{wire_s     = 7,
	       tbin_s     = 3  // for 8 time bin
  };
      /// The packed digi content  
  struct PackedDigiType {
    unsigned int wire     : wire_s;
    unsigned int tbin     : tbin_s;
  };

      /// The packed data as seen by the persistency - should never be used
      /// directly, only by calling data().
      /// Made public to be able to generate lcgdict, SA, 27/4/05
  struct PersistentPacking {
    unsigned int w1;
  };

  /// Constructors

  explicit CSCWireDigi (int wire, int tbin);  /// from the wire#, tbin#
  explicit CSCWireDigi (ChannelType channel); /// from channel
  CSCWireDigi (PackedDigiType packed_value);  /// from a packed value
  CSCWireDigi (const CSCWireDigi& digi);      /// copy
  CSCWireDigi ();                             /// default

  /// Assignment operator

  CSCWireDigi& operator=(const CSCWireDigi& digi);

  /// Gets

      /// all digi content in a packed format
  PackedDigiType packedData() const { return *(data()); }
      /// the channel identifier and the digi number packed together
  ChannelType channel() const ;
      /// return wire number
  int getWireGroup() const;
      /// return tbin number
  int getBeamCrossingTag() const;

  /// Prints

      /// Print content of digi
  void print() const;
      /// Print the binary representation of the digi
  void dump() const;

private:
  friend class testCSCDigis;

  /// Set, access, repack

  void set(int  wire, int tbin);      /// set data words
  void setData(PackedDigiType p);     /// set from a PackedDigiType
  PackedDigiType* data();             /// access to the packed data
  const PackedDigiType* data() const; /// const access to the packed data
  struct ChannelPacking {
    unsigned int wire  : wire_s;
    unsigned int tbin  : tbin_s;
  };                                 /// repack the channel number to an int

  PersistentPacking persistentData;

};

#include<iostream>
  /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCWireDigi& digi) {
  return o << " " << digi.getWireGroup()
	   << " " << digi.getBeamCrossingTag();
}
#endif
