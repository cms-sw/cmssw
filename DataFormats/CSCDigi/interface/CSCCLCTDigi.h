#ifndef CSCCLCTDigi_CSCCLCTDigi_h
#define CSCCLCTDigi_CSCCLCTDigi_h

/**\class CSCCLCTDigi
 *
 * Digi for CLCT trigger primitives. 
 *
 * $Date$
 * $Revision$
 *
 * \author N. Terentiev, CMU
 */

class CSCCLCTDigi{

public:

  typedef unsigned int ChannelType;

  /// Enum, structures

      /// Length of packed fields
  enum packing{trknmb_s    = 2, // Track number (1,2)
               pattern_s   = 3, // Pattern number (?)
               quality_s   = 2, // Quality (?)
               bend_s      = 1, // Bend (0-left, 1-right)
               striptype_s = 1, // Half- or Di- strip pattern flag (0/1)
               strip_s     = 8, // Half- or Di-  strip (0-159,0-39) in layer 3
               bx_s        = 4  // BX low order 4 bits
  };
      /// The packed digi content  
  struct PackedDigiType {
    unsigned int trknmb    : trknmb_s;
    unsigned int pattern   : pattern_s;
    unsigned int quality   : quality_s;
    unsigned int bend      : bend_s;
    unsigned int striptype : striptype_s;
    unsigned int strip     : strip_s;
    unsigned int bx        : bx_s;
  };

      /// The packed data as seen by the persistency - should never be used
      /// directly, only by calling data().
      /// Made public to be able to generate lcgdict, SA, 27/4/05
  struct PersistentPacking {
    unsigned int w1;
  };

  /// Constructors

  explicit CSCCLCTDigi (int trknmb, int pattern, int quality, int bend, int striptype, int strip, int bx);  
  explicit CSCCLCTDigi (ChannelType channel); /// from channel
  CSCCLCTDigi (PackedDigiType packed_value);  /// from a packed value
  CSCCLCTDigi (const CSCCLCTDigi& digi);      /// copy
  CSCCLCTDigi ();                             /// default

  /// Assignment operator

  CSCCLCTDigi& operator=(const CSCCLCTDigi& digi);

  /// Gets

      /// all digi content in a packed format
  PackedDigiType packedData() const { return *(data()); }
      /// the channel identifier and the digi number packed together
  ChannelType channel() const ;

      /// return track number 
  int getTrknmb()    const;
      /// return pattern number
  int getPattern()   const;
      /// return quality
  int getQuality()   const;
      /// return bend
  int getBend()      const;
      /// return striptype
  int getStriptype() const;
      /// return strip
  int getStrip()     const;
      /// return BX
  int getBx()        const;

  /// Prints

      /// Print content of digi
  void print() const;
      /// Print the binary representation of the digi
  void dump() const;

private:
  friend class testCSCDigis;

  /// Set, access, repack

  void set(int trknmb, int pattern, int quality, int bend, int striptype, int strip, int bx); /// set data words
  void setData(PackedDigiType p);     /// set from a PackedDigiType
  PackedDigiType* data();             /// access to the packed data
  const PackedDigiType* data() const; /// const access to the packed data
  struct ChannelPacking {
    unsigned int trknmb    : trknmb_s;
    unsigned int pattern   : pattern_s;
    unsigned int quality   : quality_s;
    unsigned int bend      : bend_s;
    unsigned int striptype : striptype_s;
    unsigned int strip     : strip_s;
    unsigned int bx        : bx_s;
  };                                 /// repack the channel number to an int

  PersistentPacking persistentData;

};

#include<iostream>
  /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCCLCTDigi& digi) {
  return o << " " << digi.getTrknmb()
           << " " << digi.getPattern()
           << " " << digi.getQuality()
           << " " << digi.getBend()
           << " " << digi.getStriptype()
           << " " << digi.getStrip()
	   << " " << digi.getBx();
}
#endif
