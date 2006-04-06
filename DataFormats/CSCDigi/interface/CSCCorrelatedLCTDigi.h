#ifndef CSCCorrelatedLCTDigi_CSCCorrelatedLCTDigi_h
#define CSCCorrelatedLCTDigi_CSCCorrelatedLCTDigi_h

/**\class CSCCorrelatedLCTDigi
 *
 * Digi for Correlated LCT trigger primitives. 
 *
 * $Date: 2006/03/03 22:28:47 $
 * $Revision: 1.3 $
 *
 * \author L. Gray, UF
 */

#include <boost/cstdint.hpp>

class CSCCorrelatedLCTDigi 
{
 public:
  

  /// Constructors

  explicit CSCCorrelatedLCTDigi(int trknmb, int valid, int quality,       /// from values
				int keywire, int strip, int clct_pattern, /// clct pattern is 4 bit pattern! 
				int bend, int bx);                        /// (pattern) | (strip_type << 3) 
   CSCCorrelatedLCTDigi         ();                                        /// default


  /// Gets

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
  /// return pattern 
  int getPattern() const;
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

 private:

  friend class testCSCDigis;
  uint16_t trknmb;
  uint16_t quality;
  uint16_t keywire;
  uint16_t strip;
  uint16_t pattern;
  uint16_t bend;
  uint16_t bx;
  uint16_t valid;

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
