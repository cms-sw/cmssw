#ifndef CSCCorrelatedLCTDigi_CSCCorrelatedLCTDigi_h
#define CSCCorrelatedLCTDigi_CSCCorrelatedLCTDigi_h

/**\class CSCCorrelatedLCTDigi
 *
 * Digi for Correlated LCT trigger primitives. 
 *
 * $Date:$
 * $Revision:$
 *
 * \author L. Gray, UF
 */

class CSCCorrelatedLCTDigi 
{
 public:
  
  typedef unsigned int ChannelType;

  enum packing{ trknmb_s       = 2,  // Track Number (1,2), 0 if from TF
		quality_s      = 4,  // Quality (0-15)
		wire_s         = 7,  // Key Wire		
		strip_s        = 8,  // Half/Di strip
		clct_pattern_s = 4,  // Pattern number
		bend_s         = 1,  // Bend (0-Left, 1-right)
		bx_s           = 4,  // BX 4 lsb
		valid_s        = 1   // Valid Pattern
              };

  /// The packed digi content
  struct PackedDigiType {
    unsigned int trknmb    : trknmb_s ;
    unsigned int quality   : quality_s ;
    unsigned int keywire   : wire_s ;
    unsigned int strip     : strip_s ;
    unsigned int pattern   : clct_pattern_s ;
    unsigned int bend      : bend_s ;
    unsigned int bx        : bx_s ;
    unsigned int valid     : valid_s ;
  };

  /// The packed data as seen by the persistency - should never be used
  /// directly, only by calling data().
  struct PersistentPacking {
    unsigned int w1;
  };

  /// Constructors

  explicit CSCCorrelatedLCTDigi(int trknmb, int valid, int quality,       /// from values
				int keywire, int strip, int clct_pattern, /// clct pattern is 4 bit pattern! 
				int bend, int bx);                        /// (pattern) | (strip_type << 3) 
  explicit CSCCorrelatedLCTDigi(ChannelType channel);                     /// from channel
  CSCCorrelatedLCTDigi         (PackedDigiType packed_value);             /// from packed digi
  CSCCorrelatedLCTDigi         (const CSCCorrelatedLCTDigi &digi);        /// copy
  CSCCorrelatedLCTDigi         ();                                        /// default

  /// Assignment Operator

  CSCCorrelatedLCTDigi& operator=(const CSCCorrelatedLCTDigi &digi);

  /// Gets

  /// all digi content in a packed format
  PackedDigiType packedData() const { return *(data()); }
  /// the channel identifier and the digi number packed together
  ChannelType channel() const ;

  /// return track number number
  int getTrknmb() const;
  /// return valid pattern bit
  int getValid() const;  // obsolete, use isValid()
  bool isValid() const;
  /// return the 4 bit Correlated LCT Quality
  int getQuality() const;
  /// return the key wire group
  int getKwire() const;  // obsolete, use getKeyWG()
  int getKeyWG() const;
  /// return the strip
  int getStrip() const;
  /// return CLCT pattern number
  int getCLCTPattern() const;
  /// return strip type
  int getStriptype() const; // obsolete, use getStripType()
  int getStripType() const; 
  /// return bend
  int getBend() const;
  /// return BX
  int getBx() const;        // obsolete, use getBX()
  int getBX() const;
  
  /// Prints

  /// Print content of correlated LCT digi
  void print() const;
  /// Print the binary representation of the digi.
  void dump() const;

 private:

  friend class testCSCDigis;

  void set(int trknmb, int valid, int quality,        /// set data words
	   int keywire, int strip, int clct_pattern, 
	   int bend, int bx);
  void setData(PackedDigiType p);     /// set from a PackedDigiType
  PackedDigiType* data();             /// access to the packed data
  const PackedDigiType* data() const; /// const access to the packed data
  struct ChannelPacking {
    unsigned int trknmb     : trknmb_s;
    unsigned int quality    : quality_s;
    unsigned int keywire    : wire_s;
    unsigned int strip      : strip_s;
    unsigned int pattern    : clct_pattern_s;
    unsigned int bend       : bend_s;
    unsigned int bx         : bx_s;
    unsigned int valid      : valid_s;
  }; /// repack the channel number to an int

  PersistentPacking persistentData;
};

#include<iostream>

inline std::ostream & operator<<(std::ostream & o, const CSCCorrelatedLCTDigi& digi) {
  return o << " " << digi.getTrknmb()
	   << " " << digi.isValid()
	   << " " << digi.getQuality()
	   << " " << digi.getKeyWG()
	   << " " << digi.getStrip()
           << " " << digi.getCLCTPattern()	 
	   << " " << digi.getStripType()
	   << " " << digi.getBend()
	   << " " << digi.getBX();
}
#endif
