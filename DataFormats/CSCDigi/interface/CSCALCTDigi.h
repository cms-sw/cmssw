#ifndef CSCALCTDigi_CSCALCTDigi_h
#define CSCALCTDigi_CSCALCTDigi_h

/**\class CSCALCTDigi
 *
 * Digi for ALCT trigger primitives. 
 *
 * $Date$
 * $Revision$
 *
 * \author N. Terentiev, CMU
 */

class CSCALCTDigi{

public:

  typedef unsigned int ChannelType;

  /// Enum, structures

      /// Length of packed fields
  enum packing{trknmb_s     = 2, // track number (1,2)
               keywire_s   = 7, // Key Wire group in layer 3
               bx_s        = 4, // BX
               quality_s   = 2, // Quality of pattern
               pattern_s   = 2, // Pattern number  
	       valid_s     = 1  // ALCT validity (1- valid ALCT)
  };
      /// The packed digi content  
  struct PackedDigiType {
    unsigned int trknmb     : trknmb_s;
    unsigned int keywire   : keywire_s;
    unsigned int bx        : bx_s;
    unsigned int quality   : quality_s;
    unsigned int pattern   : pattern_s;
    unsigned int valid     : valid_s;
  };

      /// The packed data as seen by the persistency - should never be used
      /// directly, only by calling data().
      /// Made public to be able to generate lcgdict, SA, 27/4/05
  struct PersistentPacking {
    unsigned int w1;
  };

  /// Constructors

  explicit CSCALCTDigi (int trknmb, int keywire,int bx, int quality, int pattern, int valid);  
  explicit CSCALCTDigi (ChannelType channel); /// from channel
  CSCALCTDigi (PackedDigiType packed_value);  /// from a packed value
  CSCALCTDigi (const CSCALCTDigi& digi);      /// copy
  CSCALCTDigi ();                             /// default

  /// Assignment operator

  CSCALCTDigi& operator=(const CSCALCTDigi& digi);

  /// Gets

      /// all digi content in a packed format
  PackedDigiType packedData() const { return *(data()); }
      /// the channel identifier and the digi number packed together
  ChannelType channel() const ;

      /// return track number number
  int getTrknmb() const;
      /// return key wire group
  int getKwire() const;
      /// return BX
  int getBx() const;
      /// return quality
  int getQuality() const;
      /// return pattern
  int getPattern() const;
      /// return ALCT validity
  int getValid() const;


  /// Prints

      /// Print content of digi
  void print() const;
      /// Print the binary representation of the digi
  void dump() const;

private:
  friend class testCSCDigis;

  /// Set, access, repack

  void set(int trknmb, int keywire,int bx, int quality, int pattern, int valid);      /// set data words
  void setData(PackedDigiType p);     /// set from a PackedDigiType
  PackedDigiType* data();             /// access to the packed data
  const PackedDigiType* data() const; /// const access to the packed data
  struct ChannelPacking {
    unsigned int trknmb     : trknmb_s;
    unsigned int keywire   : keywire_s;
    unsigned int bx        : bx_s;
    unsigned int quality   : quality_s;
    unsigned int pattern   : pattern_s;
    unsigned int valid     : valid_s;
  };                                 /// repack the channel number to an int

  PersistentPacking persistentData;

};

#include<iostream>
  /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCALCTDigi& digi) {
  return o << " " << digi.getTrknmb()
           << " " << digi.getKwire()
           << " " << digi.getBx()
           << " " << digi.getQuality()
           << " " << digi.getPattern()
	   << " " << digi.getValid();
}
#endif
