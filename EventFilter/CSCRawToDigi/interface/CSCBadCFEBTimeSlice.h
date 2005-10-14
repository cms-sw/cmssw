#ifndef CSCBadCFEBTimeSlice_h
#define CSCBadCFEBTimeSlice_h

#include "EventFilter/CSCRawToDigi/interface/CSCBadCFEBWord.h"
#include<iosfwd>

/**
 * When a time slice is bad, it only has four words, and they all start with "B"
 */


class CSCBadCFEBTimeSlice {
public:
  unsigned sizeInWords() const {return 4;}
  /// count from zero
  CSCBadCFEBWord & word(int i);

  bool check() const;


private:
  CSCBadCFEBWord theWords[4];
};

#endif
