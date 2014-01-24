#ifndef CSCDigi_CSCCorrelatedLCTDigi_h
#define CSCDigi_CSCCorrelatedLCTDigi_h

/**\class CSCCorrelatedLCTDigi
 *
 * Digi for Correlated LCT trigger primitives. 
 *
 * \author L. Gray, UF
 */

#include <boost/cstdint.hpp>
#include <iosfwd>

class CSCCorrelatedLCTDigi 
{
 public:
  
  /// Constructors
  CSCCorrelatedLCTDigi(const int trknmb, const int valid, const int quality,
		       const int keywire, const int strip, const int pattern,
		       const int bend, const int bx, const int mpclink = 0,
		       const uint16_t bx0=0, const uint16_t syncErr = 0,
		       const uint16_t cscID=0);
  CSCCorrelatedLCTDigi();                               /// default

  /// clear this LCT
  void clear();

  /// return track number
  int getTrknmb()  const { return trknmb; }

  /// return valid pattern bit
  bool isValid()   const { return valid; }

  /// return the 4 bit Correlated LCT Quality
  int getQuality() const { return quality; }

  /// return the key wire group
  int getKeyWG()   const { return keywire; }

  /// return the key halfstrip from 0,159
  int getStrip()   const { return strip; }

  /// return pattern
  int getPattern() const { return pattern; }

  /// return bend
  int getBend()    const { return bend; }

  /// return BX
  int getBX()      const { return bx; }

  /// return CLCT pattern number (in use again Feb 2011)
  int getCLCTPattern() const { return (pattern & 0xF); }

  /// return strip type (obsolete since mid-2008)
  int getStripType() const   { return ((pattern & 0x8) >> 3); }

  /// return MPC link number, 0 means not sorted, 1-3 give MPC sorting rank
  int getMPCLink() const { return mpclink; }

  uint16_t getCSCID()   const {return cscID;}
  uint16_t getBX0()     const {return bx0;}
  uint16_t getSyncErr() const {return syncErr;}

  /// Set track number (1,2) after sorting LCTs.
  void setTrknmb(const uint16_t number) {trknmb = number;}

  /// Set mpc link number after MPC sorting
  void setMPCLink(const uint16_t& link) { mpclink = link; }

  /// Print content of correlated LCT digi
  void print() const;

  ///Comparison
  bool operator == (const CSCCorrelatedLCTDigi &) const;
  bool operator != (const CSCCorrelatedLCTDigi &rhs) const
    { return !(this->operator==(rhs)); }

  /// set wiregroup number
  void setWireGroup(unsigned int wiregroup) {keywire= wiregroup;}

  /// set quality code
  void setQuality(unsigned int q) {quality=q;}

  //------- GEM matching extras ---------
  /// if has any sort of associated GEM pad
  bool hasGEM() const {return gemDPhi > -9. && gemDPhi < 9.;}

  /// LCT phi - gem pad phi for ME11
  float getGEMDPhi() const {return gemDPhi;}

  /// set gem pad deltaPhi for ME11
  void setGEMDPhi(const float dphi) {gemDPhi = dphi;}

 private:
  uint16_t trknmb;
  uint16_t valid;
  uint16_t quality;
  uint16_t keywire;
  uint16_t strip;
  uint16_t pattern;
  uint16_t bend;
  uint16_t bx;
  uint16_t mpclink;
  uint16_t bx0; 
  uint16_t syncErr;
  uint16_t cscID;
  float gemDPhi;
};

std::ostream & operator<<(std::ostream & o, const CSCCorrelatedLCTDigi& digi);

#endif
