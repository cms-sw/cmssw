#ifndef CSCCLCTDigi_CSCCLCTDigi_h
#define CSCCLCTDigi_CSCCLCTDigi_h

/**\class CSCCLCTDigi
 *
 * Digi for CLCT trigger primitives. 
 *
 * $Date:$
 * $Revision:$
 *
 * \author N. Terentiev, CMU
 */

class CSCCLCTDigi{

public:

  typedef unsigned int ChannelType;

  /// Enum, structures

      /// Length of packed fields
  enum packing{
                valid_s    = 1, // CLCT validity (1 - valid CLCT)
                quality_s  = 3, // Quality of pattern (Hits on pattern)
                patshape_s = 3, // Pattern shape
                striptype_s= 1, // Strip Type 1=Half-Strip, 0=Di-Strip
                bend_s     = 1, // Bend direction (0-left, 1-right)
                strip_s    = 7, // Key Half-strip,Di-Strip points to lower Half
                cfeb_s     = 3, // Key CFEB ID
                bx_s       = 2, // BXN
                trknmb_s   = 2  // Track number (1,2)
  };
      /// The packed digi content  
  struct PackedDigiType {
    unsigned int valid     : valid_s;
    unsigned int quality   : quality_s;
    unsigned int patshape  : patshape_s; 
    unsigned int striptype : striptype_s;
    unsigned int bend      : bend_s;
    unsigned int strip     : strip_s;
    unsigned int cfeb      : cfeb_s;
    unsigned int bx        : bx_s;
    unsigned int trknmb    : trknmb_s;
 };

      /// The packed data as seen by the persistency - should never be used
      /// directly, only by calling data().
      /// Made public to be able to generate lcgdict, SA, 27/4/05
  struct PersistentPacking {
    unsigned int w1;
  };

  /// Constructors

  explicit CSCCLCTDigi (int valid, int quality, int patshape, int striptype, int bend,  int strip, int cfeb, int bx, int trknmb);  
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

  int getValid()     const;     // return validity
  int getQuality()   const;     // return quality
  int getPattern()   const;     // return pattern shape
  int getStriptype() const;     // return striptype
  int getBend()      const;     // return bend
  int getStrip()     const;     // return strip
  int getCfeb ()     const;     // return Key CFEB ID
  int getBx()        const;     // return BX
  int getTrknmb()    const;     // return track number

  /// Prints

      /// Print content of digi
  void print() const;
      /// Print the binary representation of the digi
  void dump() const;

private:
  friend class testCSCDigis;

  /// Set, access, repack

  void set(int valid, int quality, int patshape, int striptype, int bend,  int strip, int cfeb, int bx, int trknmb); /// set data words
  void setData(PackedDigiType p);     /// set from a PackedDigiType
  PackedDigiType* data();             /// access to the packed data
  const PackedDigiType* data() const; /// const access to the packed data
  struct ChannelPacking {
    unsigned int valid     : valid_s;
    unsigned int quality   : quality_s;
    unsigned int patshape  : patshape_s;
    unsigned int striptype : striptype_s;
    unsigned int bend      : bend_s;
    unsigned int strip     : strip_s;
    unsigned int cfeb      : cfeb_s;
    unsigned int bx        : bx_s;
    unsigned int trknmb    : trknmb_s;
  };                                 /// repack the channel number to an int

  PersistentPacking persistentData;

};

#include<iostream>
  /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCCLCTDigi& digi) {
  return o << " " << digi.getValid()
           << " " << digi.getQuality()
           << " " << digi.getPattern()
           << " " << digi.getStriptype()
           << " " << digi.getBend()
           << " " << digi.getStrip()
           << " " << digi.getCfeb()
	   << " " << digi.getBx()
           << " " << digi.getTrknmb();
}
#endif
