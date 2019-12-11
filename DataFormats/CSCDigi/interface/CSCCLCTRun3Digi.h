#ifndef CSCDigi_CSCCLCTRun3Digi_h
#define CSCDigi_CSCCLCTRun3Digi_h

#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include <vector>

class CSCCLCTRun3Digi : public CSCCLCTDigi {
public:
  typedef std::vector<std::vector<uint16_t>> ComparatorContainer;

  /// Constructors
  CSCCLCTRun3Digi(const int valid,
                  const int quality,
                  const int pattern,
                  const int striptype,
                  const int bend,
                  const int strip,
                  const int cfeb,
                  const int bx,
                  const int compCode,
                  const int trknmb = 0,
                  const int fullbx = 0);
  /// default
  CSCCLCTRun3Digi();

  /// clear this CLCT
  void clear();

  /// Convert strip_ and cfeb_ to keyStrip. Each CFEB has up to
  /// 32 half-strips or 64 quart-strips.
  /// There are up to 7 cfebs.  The "strip_" variable is one
  /// of 32 half-strips or 64 quart-strips on the
  /// keylayer of a single CFEB, so that:
  /// - half-strip  = (cfeb * 32 + strip)
  /// - quart-strip = (cfeb * 64 + strip).
  /// Return the corresponding half-strip/quart-strip number
  /// depending on whether the fit to comparator digis was done.
  int getKeyStrip(bool doFit = true) const;

  // 12-bit comparator code
  int getCompCode() const { return compCode_; }

  void setCompCode(const int16_t code) { compCode_ = code; }

  // comparator hits in this CLCT
  ComparatorContainer getHits() const { return hits_; }

  void setHits(const ComparatorContainer& hits) { hits_ = hits; }

  /// True if the two LCTs have exactly the same members (except the number).
  bool operator==(const CSCCLCTRun3Digi&) const;

  /// Print content of digi.
  void print() const;

private:
  // 12-bit comparator code
  uint16_t compCode_;

  // which hits are in this CLCT?
  ComparatorContainer hits_;
};

std::ostream& operator<<(std::ostream& o, const CSCCLCTRun3Digi& digi);

#endif
