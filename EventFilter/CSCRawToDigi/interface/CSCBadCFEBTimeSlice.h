#ifndef EventFilter_CSCRawToDigi_CSCBadCFEBTimeSlice_h
#define EventFilter_CSCRawToDigi_CSCBadCFEBTimeSlice_h

#include "EventFilter/CSCRawToDigi/interface/CSCBadCFEBWord.h"
#include <iosfwd>

/**
 * When a time slice is bad, it only has four words, and they all start with "B"
 */

class CSCBadCFEBTimeSlice {
public:
  unsigned sizeInWords() const { return 4; }
  /// count from zero
  const CSCBadCFEBWord& word(int i) const;

  bool check() const;

private:
  CSCBadCFEBWord theWords[4];
};

#endif
