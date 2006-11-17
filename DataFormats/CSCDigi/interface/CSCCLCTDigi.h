#ifndef CSCDigi_CSCCLCTDigi_h
#define CSCDigi_CSCCLCTDigi_h

/**\class CSCCLCTDigi
 *
 * Digi for CLCT trigger primitives. 
 *
 * $Date: 2006/06/20 10:26:42 $
 * $Revision: 1.8 $
 *
 * \author N. Terentiev, CMU
 */

#include <boost/cstdint.hpp>

class CSCCLCTDigi {

 public:

  /// Constructors
  CSCCLCTDigi(const int valid, const int quality, const int pattern,
	      const int striptype, const int bend, const int strip,
	      const int cfeb, const int bx, const int trknmb = 0);
  /// default
  CSCCLCTDigi();

  /// clear this CLCT
  void clear();

  /// check CLCT validity (1 - valid CLCT)
  bool isValid()     const {return valid_ ;}

  /// return quality of a pattern (number of layers hit!)
  int getQuality()   const {return quality_ ;}

  /// return pattern
  int getPattern()   const {return pattern_ ;}

  /// return striptype
  int getStripType() const {return striptype_ ;}

  /// return bend
  int getBend()      const {return bend_ ;}

  /// return strip
  int getStrip()     const {return strip_ ;}

  /// return Key CFEB ID
  int getCFEB()      const {return cfeb_ ;}

  /// return BX
  int getBX()        const {return bx_ ;}

  /// return track number (1,2)
  int getTrknmb()    const {return trknmb_ ;}

  /// Convert strip_ and cfeb_ to keyStrip. Each CFEB has up to 16 strips
  /// (32 halfstrips). There are 5 cfebs.  The "strip_" variable is one
  /// of 32 halfstrips on the keylayer of a single CFEB, so that
  /// Distrip   = (cfeb*32 + strip)/4.
  /// Halfstrip = (cfeb*32 + strip).
  /// Always return halfstrip number since this is what is stored in
  /// the correlated LCT digi.  For distrip patterns, the convention is
  /// the same as for persistent strip numbers: low halfstrip of a distrip.
  /// SV, June 15th, 2006.
  int getKeyStrip()  const {
    int keyStrip = cfeb_ * 32 + strip_;
    return keyStrip;
  }

  /// Set track number (1,2) after sorting CLCTs.
  void setTrknmb(const uint16_t number) {trknmb_ = number;}

  /// True if the left-hand side has a larger "quality".  Full definition
  /// of "quality" depends on quality word itself, pattern type, and strip
  /// number.
  bool operator >  (const CSCCLCTDigi&) const;

  /// True if the two LCTs have exactly the same members OR they
  /// are high- and low-pT LCTs from the exact same location in the chamber.
  bool operator == (const CSCCLCTDigi&) const;

  /// False only when both LCTs have exactly the same members.
  /** @@ Shouldn't be false only when the preceding one is true?
      To be checked. */
  bool operator != (const CSCCLCTDigi&) const;

  /// Print content of digi.
  void print() const;

 private:

  uint16_t valid_      ;
  uint16_t quality_    ;
  uint16_t pattern_    ;
  uint16_t striptype_  ;
  uint16_t bend_       ;
  uint16_t strip_      ;
  uint16_t cfeb_       ;
  uint16_t bx_         ;
  uint16_t trknmb_     ;
};

#include<iostream>
inline std::ostream & operator<<(std::ostream & o, const CSCCLCTDigi& digi) {
  return o << "CSC CLCT #"    << digi.getTrknmb()
	   << ": Valid = "    << digi.isValid()
           << " Quality = "   << digi.getQuality()
           << " Pattern = "   << digi.getPattern()
           << " StripType = " << digi.getStripType()
           << " Bend = "      << digi.getBend()
           << " Strip = "     << digi.getStrip()
           << " KeyStrip = "  << digi.getKeyStrip()
           << " CFEB = "      << digi.getCFEB()
	   << " BX = "        << digi.getBX();
}
#endif
