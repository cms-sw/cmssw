#ifndef CSCDigi_CSCCorrelatedLCTDigi_h
#define CSCDigi_CSCCorrelatedLCTDigi_h

/**\class CSCCorrelatedLCTDigi
 *
 * Digi for Correlated LCT trigger primitives. 
 *
 *
 * \author L. Gray, UF
 */

#include <cstdint>
#include <iosfwd>
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigi.h"

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

  /// return the key wire group. counts from 0.
  int getKeyWG()   const { return keywire; }

  /// return the key halfstrip from 0,159
  int getStrip()   const { return strip; }

  /// return the fractional strip. counts from 0.25
  float getFractionalStrip() const;

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

  /// set valid
  void setValid(unsigned int v) {valid=v;}

  /// set strip
  void setStrip(unsigned int s) {strip=s;}

  /// set pattern
  void setPattern(unsigned int p) {pattern=p;}

  /// set bend
  void setBend(unsigned int b) {bend=b;}

  /// set bx
  void setBX(unsigned int b) {bx=b;}

  /// set bx0
  void setBX0(unsigned int b) {bx0=b;}

  /// set syncErr
  void setSyncErr(unsigned int s) {syncErr=s;}

  /// set cscID
  void setCSCID(unsigned int c) {cscID=c;}

  /// SIMULATION ONLY ////
  enum Type{CLCTALCT, // CLCT-centric
	    ALCTCLCT, // ALCT-centric
	    ALCTCLCTGEM, // ALCT-CLCT-1 GEM pad
	    ALCTCLCT2GEM, // ALCT-CLCT-2 GEM pads in coincidence
	    ALCT2GEM, // ALCT-2 GEM pads in coincidence
	    CLCT2GEM  // CLCT-2 GEM pads in coincidence
  };
  
  int getType() {return type_;}
  void setType(int type) {type_ = type;}
  
  void setALCT(const CSCALCTDigi& alct) {alct_ = alct;}
  void setCLCT(const CSCCLCTDigi& clct) {clct_ = clct;}
  void setGEM1(const GEMPadDigi& gem) {gem1_ = gem;}
  void setGEM2(const GEMPadDigi& gem) {gem2_ = gem;}
  const CSCALCTDigi&  getALCT() {return alct_;}
  const CSCCLCTDigi&  getCLCT() {return clct_;}
  const GEMPadDigi&  getGEM1() {return gem1_;}
  const GEMPadDigi&  getGEM2() {return gem2_;}

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

  /// SIMULATION ONLY ////
  int type_;

  CSCALCTDigi alct_;
  CSCCLCTDigi clct_;
  GEMPadDigi gem1_;
  GEMPadDigi gem2_;
};

std::ostream & operator<<(std::ostream & o, const CSCCorrelatedLCTDigi& digi);

#endif
