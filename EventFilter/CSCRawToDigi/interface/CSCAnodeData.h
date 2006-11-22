#ifndef CSCAnodeData_h
#define CSCAnodeData_h
#include <vector>
#include <cassert>
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
class CSCALCTHeader;

#include <iostream> // get rid of when debug is gone
class CSCAnodeDataFrame {
public:
  CSCAnodeDataFrame() {}
  CSCAnodeDataFrame(unsigned chip, unsigned tbin, unsigned data) 
  :  data_(data), tbin_(tbin),
  // do we have 2 bits for this, or 3? 
  // chip_( chip&1 + 2*(chip&2 | chip&4) ) 
  chip_(chip) {}

  /// given a wiregroup between 0 and 7, it tells whether this bit was on
  bool isHit(unsigned wireGroup) const {
    assert(wireGroup < 8);
    return ( (data_>>wireGroup) & 0x1 );
  }

  /// sets a bit, from 0 to 7
  void addHit(unsigned wireGroup) {
    data_ |= (1 << wireGroup);
  }

  /// time bin
  unsigned tbin() const {return tbin_;}
  /// kind of the chip ID.  But it's only 2-bit, and we really need
  /// three, so it's the lowest bit, plus the OR of the next two.
  unsigned chip() const {return chip_;}
  unsigned short data() const {return data_;}
private:
  unsigned short data_ : 8;
  unsigned short tbin_ : 5;
  unsigned short chip_ : 3;
  //unsigned short ddu_code_ : 1;
};


class CSCAnodeData {

public:
  CSCAnodeData();
  /// a blank one, for Monte Carlo
  CSCAnodeData(const CSCALCTHeader &);
  /// fill from a real datastream
  CSCAnodeData(const CSCALCTHeader &, const unsigned short *buf);

/** turns on the debug flag for this class */
  static void setDebug(){debug = true;};

/** turns off the debug flag for this class (default) */
  static void setNoDebug(){debug = false;};

  unsigned short * data() {return theDataFrames;}
  /// the amount of the input binary buffer read, in 16-bit words
  int sizeInWords() const {return nFrames();}

  /// the number of data frames
  int nFrames() const {return nAFEBs_ * nTimeBins_ * 6 * 2;}
 
  /** turns on/off debug flag for this class */
  static void setDebug(bool value) {debug = value;};

  /// input layer is from 1 to 6
  std::vector<CSCWireDigi> wireDigis(int layer) const;
  std::vector<std::vector<CSCWireDigi> > wireDigis() const;

  const CSCAnodeDataFrame & rawHit(int afeb, int tbin, int layer, int halfLayer) const {
    return (const CSCAnodeDataFrame &)(theDataFrames[index(afeb, tbin, layer)+halfLayer]);
  }

  /// nonconst version
  CSCAnodeDataFrame & rawHit(int afeb, int tbin, int layer, int halfLayer) {
    return (CSCAnodeDataFrame &)(theDataFrames[index(afeb, tbin, layer)+halfLayer]); 
  }

  void add(const CSCWireDigi &, int layer);

  static bool selfTest();

private:
  static bool debug;
  /// the index into theDataFrames
  int index(int afeb, int tbin, int layer) const {
    int result = (layer-1)*2 + 12*tbin + afeb*12*nTimeBins_;
    assert(result < nFrames());
    return result;
  }

  /// we don't know the size at first.  Max should be 7 boards * 32 bins * 6 layers * 2
  unsigned short theDataFrames[2700];
  int nAFEBs_;
  int nTimeBins_;
};

#endif


