#ifndef DataFormats_CSCDigi_CSCCorrelatedLCTDigi_h
#define DataFormats_CSCDigi_CSCCorrelatedLCTDigi_h

/**\class CSCCorrelatedLCTDigi
 *
 * Digi for Correlated LCT trigger primitives.
 *
 *
 * \author L. Gray, UF
 */

#include <cstdint>
#include <iosfwd>
#include <limits>
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigi.h"

class CSCCorrelatedLCTDigi {
public:
  enum LCTKeyStripMasks { kEightStripMask = 0x1, kQuartStripMask = 0x1, kHalfStripMask = 0xff };
  enum LCTKeyStripShifts { kEightStripShift = 9, kQuartStripShift = 8, kHalfStripShift = 0 };
  // temporary to facilitate CCLUT-EMTF/OMTF integration studies
  enum LCTPatternMasks { kRun3SlopeMask = 0xf, kRun3PatternMask = 0x7, kLegacyPatternMask = 0xf };
  enum LCTPatternShifts { kRun3SlopeShift = 7, kRun3PatternShift = 4, kLegacyPatternShift = 0 };
  enum class Version { Legacy = 0, Run3 };

  /// Constructors
  CSCCorrelatedLCTDigi(const uint16_t trknmb,
                       const uint16_t valid,
                       const uint16_t quality,
                       const uint16_t keywire,
                       const uint16_t strip,
                       const uint16_t pattern,
                       const uint16_t bend,
                       const uint16_t bx,
                       const uint16_t mpclink = 0,
                       const uint16_t bx0 = 0,
                       const uint16_t syncErr = 0,
                       const uint16_t cscID = 0,
                       const uint16_t hmt = 0,
                       const Version version = Version::Legacy);
  /// default
  CSCCorrelatedLCTDigi();

  /// clear this LCT
  void clear();

  /// return track number
  uint16_t getTrknmb() const { return trknmb; }

  /// return valid pattern bit
  bool isValid() const { return valid; }

  /// return the Quality
  uint16_t getQuality() const { return quality; }

  /// return the key wire group. counts from 0.
  uint16_t getKeyWG() const { return keywire; }

  /// return the key halfstrip from 0,159
  uint16_t getStrip(uint16_t n = 2) const;

  /// set single quart strip bit
  void setQuartStrip(const bool quartStrip);

  /// get single quart strip bit
  bool getQuartStrip() const;

  /// set single eight strip bit
  void setEightStrip(const bool eightStrip);

  /// get single eight strip bit
  bool getEightStrip() const;

  /*
    Strips are numbered starting from 1 in CMSSW
    Half-strips, quarter-strips and eighth-strips are numbered starting from 0
    The table below shows the correct numbering
    ---------------------------------------------------------------------------------
    strip     |               1               |                 2                   |
    ---------------------------------------------------------------------------------
    1/2-strip |       0       |       1       |       2         |         3         |
    ---------------------------------------------------------------------------------
    1/4-strip |   0   |   1   |   2   |   3   |   4   |    5    |    6    |    7    |
    ---------------------------------------------------------------------------------
    1/8-strip | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
    ---------------------------------------------------------------------------------

    Note: the CSC geometry also has a strip offset of +/- 0.25 strips. When comparing the
    CLCT/LCT position with the true muon position, take the offset into account!
   */
  float getFractionalStrip(uint16_t n = 2) const;

  /// Legacy: return pattern ID
  /// Run-3: return the bending angle value
  uint16_t getPattern() const;

  /// return pattern
  uint16_t getRun3Pattern() const;

  /// return the slope
  uint16_t getSlope() const;

  /// return left/right bending
  uint16_t getBend() const { return bend; }

  /// return BX
  uint16_t getBX() const { return bx; }

  /// return CLCT pattern number (in use again Feb 2011)
  /// This function should not be used for Run-3
  uint16_t getCLCTPattern() const;

  /// return strip type (obsolete since mid-2008)
  uint16_t getStripType() const { return ((pattern & 0x8) >> 3); }

  /// return MPC link number, 0 means not sorted, 1-3 give MPC sorting rank
  uint16_t getMPCLink() const { return mpclink; }

  uint16_t getCSCID() const { return cscID; }
  uint16_t getBX0() const { return bx0; }
  uint16_t getSyncErr() const { return syncErr; }

  /// Run-3 introduces high-multiplicity bits for CSCs.
  /// The allocation is different for ME1/1 and non-ME1/1
  /// chambers. Both LCTs in a chamber are needed for the complete
  /// high-multiplicity trigger information
  uint16_t getHMT() const;

  /// Set track number (1,2) after sorting LCTs.
  void setTrknmb(const uint16_t number) { trknmb = number; }

  /// Set mpc link number after MPC sorting
  void setMPCLink(const uint16_t& link) { mpclink = link; }

  /// Print content of correlated LCT digi
  void print() const;

  ///Comparison
  bool operator==(const CSCCorrelatedLCTDigi&) const;
  bool operator!=(const CSCCorrelatedLCTDigi& rhs) const { return !(this->operator==(rhs)); }

  /// set wiregroup number
  void setWireGroup(const uint16_t wiregroup) { keywire = wiregroup; }

  /// set quality code
  void setQuality(const uint16_t q) { quality = q; }

  /// set valid
  void setValid(const uint16_t v) { valid = v; }

  /// set strip
  void setStrip(const uint16_t s) { strip = s; }

  /// set pattern
  void setPattern(const uint16_t p);

  /// set pattern
  void setRun3Pattern(const uint16_t pattern);

  /// set the slope
  void setSlope(const uint16_t slope);

  /// set bend
  void setBend(const uint16_t b) { bend = b; }

  /// set bx
  void setBX(const uint16_t b) { bx = b; }

  /// set bx0
  void setBX0(const uint16_t b) { bx0 = b; }

  /// set syncErr
  void setSyncErr(const uint16_t s) { syncErr = s; }

  /// set cscID
  void setCSCID(const uint16_t c) { cscID = c; }

  /// set high-multiplicity bits
  void setHMT(const uint16_t h);

  /// Distinguish Run-1/2 from Run-3
  bool isRun3() const { return version_ == Version::Run3; }

  void setRun3(const bool isRun3);

  /// SIMULATION ONLY ////
  enum Type {
    CLCTALCT,      // CLCT-centric
    ALCTCLCT,      // ALCT-centric
    ALCTCLCTGEM,   // ALCT-CLCT-1 GEM pad
    ALCTCLCT2GEM,  // ALCT-CLCT-2 GEM pads in coincidence
    ALCT2GEM,      // ALCT-2 GEM pads in coincidence
    CLCT2GEM,      // CLCT-2 GEM pads in coincidence
    CLCTONLY,      // Missing ALCT
    ALCTONLY       // Missing CLCT
  };

  int getType() const { return type_; }

  void setType(int type) { type_ = type; }

  void setALCT(const CSCALCTDigi& alct) { alct_ = alct; }
  void setCLCT(const CSCCLCTDigi& clct) { clct_ = clct; }
  void setGEM1(const GEMPadDigi& gem) { gem1_ = gem; }
  void setGEM2(const GEMPadDigi& gem) { gem2_ = gem; }
  const CSCALCTDigi& getALCT() const { return alct_; }
  const CSCCLCTDigi& getCLCT() const { return clct_; }
  const GEMPadDigi& getGEM1() const { return gem1_; }
  const GEMPadDigi& getGEM2() const { return gem2_; }

private:
  void setDataWord(const uint16_t newWord, uint16_t& word, const unsigned shift, const unsigned mask);
  uint16_t getDataWord(const uint16_t word, const unsigned shift, const unsigned mask) const;

  // Note: The Run-3 data format is substantially different than the
  // Run-1/2 data format. Some explanation is provided below. For
  // more information, please check "DN-20-016".

  // Run-1, Run-2 and Run-3 trknmb is either 1 or 2.
  uint16_t trknmb;
  // In Run-3, the valid will be encoded as a quality
  // value "000" or "00".
  uint16_t valid;
  // In Run-3, the LCT quality number will be 2 or 3 bits
  // For ME1/1 chambers: 3 bits
  // For non-ME1/1 chambers: 2 bits
  uint16_t quality;
  // 7-bit key wire
  uint16_t keywire;
  // In Run-3, the strip number receives two additional bits
  // strip[7:0] -> 1/2 strip value
  // strip[8]   -> 1/4 strip bit
  // strip[9]   -> 1/8 strip bit
  uint16_t strip;
  // In Run-3, the 4-bit pattern number is reinterpreted as the
  // 4-bit bending value. There will be 16 bending values * 2 (left/right)
  uint16_t pattern;
  // Common definition for left/right bending in Run-1, Run-2 and Run-3.
  // 0: right; 1: left
  uint16_t bend;
  uint16_t bx;
  uint16_t mpclink;
  uint16_t bx0;
  // The synchronization bit is actually not used by MPC or EMTF
  uint16_t syncErr;
  // 4-bit CSC chamber identifier
  uint16_t cscID;
  // In Run-3, LCT data will be carrying the high-multiplicity bits
  // for chamber. These bits may indicate the observation of "exotic" events
  // Depending on the chamber type 2 or 3 bits will be repurposed
  // in the 32-bit LCT data word from the synchronization bit and
  // quality bits.
  uint16_t hmt;

  /// SIMULATION ONLY ////
  int type_;

  CSCALCTDigi alct_;
  CSCCLCTDigi clct_;
  GEMPadDigi gem1_;
  GEMPadDigi gem2_;

  Version version_;
};

std::ostream& operator<<(std::ostream& o, const CSCCorrelatedLCTDigi& digi);

#endif
