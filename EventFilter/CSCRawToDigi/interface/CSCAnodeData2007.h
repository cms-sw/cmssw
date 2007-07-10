#ifndef CSCAnodeData2007_h
#define CSCAnodeData2007_h
#include <vector>
#include <cassert>
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
class CSCALCTHeader2007;

#include <iostream> // get rid of when debug is gone
class CSCAnodeDataFrame2007 {
public:
  CSCAnodeDataFrame2007() {}
  CSCAnodeDataFrame2007(unsigned data) 
  :  data_(data) {}

  /// given a wiregroup between 0 and 11, it tells whether this bit was on
  bool isHit(unsigned wireGroup) const {
    assert(wireGroup < 12);
    return ( (data_>>wireGroup) & 0x1 );
  }

  /// for each layer wiregroups are divided in parts each containing 12 wiregroups
  /// thus for ALCT-672 there are 112 wiregroups per layer and they are grouped into 10 parts
  unsigned short data() const {return data_;}
private:
  unsigned short data_ : 12;
  unsigned short reserved: 4;
};


class CSCAnodeData2007 {

public:
  CSCAnodeData2007();
  /// a blank one, for Monte Carlo
  CSCAnodeData2007(const CSCALCTHeader2007 &);
  /// fill from a real datastream
  CSCAnodeData2007(const CSCALCTHeader2007 &, const unsigned short *buf);

/** turns on the debug flag for this class */
  static void setDebug(){debug = true;};

/** turns off the debug flag for this class (default) */
  static void setNoDebug(){debug = false;};

  unsigned short * data() {return theDataFrames;}
  /// the amount of the input binary buffer read, in 16-bit words
  int sizeInWords() const {return nFrames();}

  /// the number of data frames 
  int nFrames() const {return nTimeBins_ * 6 * nWireGroupParts() ;}
 
  int nWireGroupParts() const;

  /** turns on/off debug flag for this class */
  static void setDebug(bool value) {debug = value;};

  /// input layer is from 1 to 6
  std::vector<CSCWireDigi> wireDigis(int layer) const;
  std::vector<std::vector<CSCWireDigi> > wireDigis() const;

  //const CSCAnodeData2007Frame & rawHit(int afeb, int tbin, int layer, int halfLayer) const {
  //  return (const CSCAnodeData2007Frame &)(theDataFrames[index(afeb, tbin, layer)+halfLayer]);
  //}

  /// nonconst version
  //CSCAnodeData2007Frame & rawHit(int afeb, int tbin, int layer, int halfLayer) {
  //  return (CSCAnodeData2007Frame &)(theDataFrames[index(afeb, tbin, layer)+halfLayer]); 
  //}

  static bool selfTest();

private:
  static bool debug;
  /// the index into theDataFrames
  //int index(int afeb, int tbin, int layer) const {
  //  int result = (layer-1)*2 + 12*tbin + afeb*12*nTimeBins_;
  //  assert(result < nFrames());
  //  return result;
  //}

  /// we don't know the size at first.  Max should be 672 wiregroups * 32 tbins 
  unsigned short theDataFrames[22000];
  int nAFEBs_;
  int nTimeBins_;
};

#endif


