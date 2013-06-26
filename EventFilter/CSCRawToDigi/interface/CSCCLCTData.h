#ifndef CSCCLCTData_h
#define CSCCLCTData_h
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include <vector>
#include <cassert>

struct CSCCLCTDataWord {
  CSCCLCTDataWord(unsigned cfeb, unsigned tbin, unsigned data)
  : data_(data), tbin_(tbin), cfeb_(cfeb) {}
  bool value(int distrip) {return (data_ >> distrip) & 0x1;}
  ///@@ not right! doesn't set zero
  void set(int distrip, bool value) {data_ |= (value << distrip);}
  unsigned short data_ : 8;
  unsigned short tbin_ : 4;
  unsigned short cfeb_ : 4;
};

class CSCTMBHeader;

class CSCCLCTData {

public:

  explicit CSCCLCTData(const CSCTMBHeader * tmbHeader);
  CSCCLCTData(int ncfebs, int ntbins);
  CSCCLCTData(int ncfebs, int ntbins, const unsigned short *e0bbuf);

  /** turns on/off debug flag for this class */
  static void setDebug(const bool value) {debug = value;};

  /// layers count from one
  std::vector<CSCComparatorDigi> comparatorDigis(int layer);

  /// layers count from one
  std::vector<CSCComparatorDigi> comparatorDigis(uint32_t idlayer, unsigned icfeb);


  unsigned short * data() {return theData;}
  /// in 16-bit words
  int sizeInWords() const { return size_;}
  int nlines() const { return ncfebs_*ntbins_*6; }

  ///TODO for packing.  Doesn't do flipping yet
  void add(const CSCComparatorDigi & digi, int layer);

  CSCCLCTDataWord & dataWord(int iline) const {
#ifdef ASSERTS
    assert(iline < nlines());
#endif
    union dataPtr { const unsigned short * s; CSCCLCTDataWord * d; } mptr;
    mptr.s = theData+iline;
    return *(mptr.d);
  }

  CSCCLCTDataWord & dataWord(int cfeb, int tbin, int layer) const {
    int iline = (layer-1) + tbin*6 + cfeb*6*ntbins_;
    return dataWord(iline);
  }

  bool bitValue(int cfeb, int tbin, int layer, int distrip) {
    return dataWord(cfeb, tbin, layer).value(distrip);
  }

  // checks that the CFEB number and time bins are correct
  bool check() const;

  // hex dump
  void dump() const;

  // checks packing and unpacking
  static void selfTest();


 private:

  // helper for constructors
  void zero();

  static bool debug;
  int ncfebs_;
  int ntbins_;
  int size_;
  unsigned short theData[5*6*32];
};

#endif
