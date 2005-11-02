#ifndef CSCRPCDigi_CSCRPCDigi_h
#define CSCRPCDigi_CSCRPCDigi_h

/**\class CSCRPCDigi
 *
 * Digi for RPC data coming thru RAT-ALCT-DDU. 
 *
 * $Date$
 * $Revision$
 *
 * \author N. Terentiev, CMU
 */

class CSCRPCDigi{

public:

  typedef unsigned int ChannelType;

  /// Enum, structures

      /// Length of packed fields
  enum packing{strip_s     = 7,
	       tbin_s      = 2  // for 4 time bins
  };
      /// The packed digi content  
  struct PackedDigiType {
    unsigned int strip     : strip_s;
    unsigned int tbin      : tbin_s;
  };

      /// The packed data as seen by the persistency - should never be used
      /// directly, only by calling data().
      /// Made public to be able to generate lcgdict, SA, 27/4/05
  struct PersistentPacking {
    unsigned int w1;
  };

  /// Constructors

  explicit CSCRPCDigi (int strip, int tbin);  /// from the strip#, tbin#
  explicit CSCRPCDigi (ChannelType channel); /// from channel
  CSCRPCDigi (PackedDigiType packed_value);  /// from a packed value
  CSCRPCDigi (const CSCRPCDigi& digi);      /// copy
  CSCRPCDigi ();                             /// default

  /// Assignment operator

  CSCRPCDigi& operator=(const CSCRPCDigi& digi);

  /// Gets

      /// all digi content in a packed format
  PackedDigiType packedData() const { return *(data()); }
      /// the channel identifier and the digi number packed together
  ChannelType channel() const ;
      /// return strip number
  int getStrip() const;
      /// return tbin number
  int getBx() const;

  /// Prints

      /// Print content of digi
  void print() const;
      /// Print the binary representation of the digi
  void dump() const;

private:
  friend class testCSCDigis;

  /// Set, access, repack

  void set(int strip, int tbin);      /// set data words
  void setData(PackedDigiType p);     /// set from a PackedDigiType
  PackedDigiType* data();             /// access to the packed data
  const PackedDigiType* data() const; /// const access to the packed data
  struct ChannelPacking {
    unsigned int strip  : strip_s;
    unsigned int tbin  : tbin_s;
  };                                 /// repack the channel number to an int

  PersistentPacking persistentData;

};

#include<iostream>
  /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCRPCDigi& digi) {
  return o << " " << digi.getStrip()
	   << " " << digi.getBx();
}
#endif
