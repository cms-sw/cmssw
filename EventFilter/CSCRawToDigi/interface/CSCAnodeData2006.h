#ifndef CSCAnodeData2006_h
#define CSCAnodeData2006_h
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeDataFormat.h"
#include <cassert>
class CSCALCTHeader;

class CSCAnodeDataFrame2006 {
public:
  CSCAnodeDataFrame2006() {}
  CSCAnodeDataFrame2006(unsigned chip, unsigned tbin, unsigned data) 
  :  data_(data&0x3), tbin_(tbin),
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
  unsigned short chip_ : 2;
  unsigned short ddu_code_ : 1;
};


class CSCAnodeData2006 : public CSCAnodeDataFormat 
{
public:
  /// a blank one, for Monte Carlo
  CSCAnodeData2006(const CSCALCTHeader &);
  /// fill from a real datastream
  CSCAnodeData2006(const CSCALCTHeader &, const unsigned short *buf);

  virtual unsigned short * data() {return theDataFrames;}
  /// the amount of the input binary buffer read, in 16-bit words
  virtual unsigned short int sizeInWords() const {return nAFEBs_ * nTimeBins_ * 6 * 2;}

  /// input layer is from 1 to 6
  virtual std::vector<CSCWireDigi> wireDigis(int layer) const;

  virtual void add(const CSCWireDigi &, int layer);

private:
  void init();

  const CSCAnodeDataFrame2006 & rawHit(int afeb, int tbin, int layer, int halfLayer) const {
    return (const CSCAnodeDataFrame2006 &)(theDataFrames[index(afeb, tbin, layer)+halfLayer]);
  }

  /// nonconst version
  CSCAnodeDataFrame2006 & rawHit(int afeb, int tbin, int layer, int halfLayer) {
    return (CSCAnodeDataFrame2006 &)(theDataFrames[index(afeb, tbin, layer)+halfLayer]);
  }

  /// the index into theDataFrames
  int index(int afeb, int tbin, int layer) const {
    int result = (layer-1)*2 + 12*tbin + afeb*12*nTimeBins_;
    assert(result < sizeInWords());
    return result;
  }

  /// we don't know the size at first.  Max should be 7 boards * 32 bins * 6 layers * 2
  unsigned short theDataFrames[2700];
  /// in 2007 format the max number of frames is 1860
  int nAFEBs_;
  int nTimeBins_;
};

#endif


