#ifndef CSCCFEBData_h
#define CSCCFEBData_h

class CSCCFEBTimeSlice;
class CSCStripDigi;
class CSCCFEBStatusDigi;

#include<vector>
#include<iosfwd>
#include<iostream>
//#include <boost/cstdint.hpp>


class CSCCFEBData {
 public:
 /// read from an existing data stream. 
  CSCCFEBData(unsigned boardNumber, unsigned short * buf);
  /// create, 
  CSCCFEBData(unsigned boardNumber, bool sixteenSamples);
  
  unsigned nTimeSamples() const { return theNumberOfSamples;}

  /// count from 0.  User should check if it's a bad slice
  const CSCCFEBTimeSlice * timeSlice(unsigned i) const;

  unsigned adcCounts(unsigned layer, unsigned channel, unsigned timeBin) const;
  unsigned adcOverflow(unsigned layer, unsigned channel, unsigned timeBin) const;
  unsigned controllerData(unsigned uglay, unsigned ugchan, unsigned timeBin) const;
  unsigned overlappedSampleFlag(unsigned layer, unsigned channel, unsigned timeBin) const;
  unsigned errorstat(unsigned layer, unsigned channel, unsigned timeBin) const;
  
  void add(const CSCStripDigi &, int layer);
  /// Fill strip digis for layer with raw detid = idlayer
  /// WARNING: these digis have no comparator information.

  ///faster way to get to digis
  void digis(uint32_t idlayer,  std::vector<CSCStripDigi> & result);

  std::vector<CSCStripDigi> digis(unsigned idlayer);
  /// deprecated.  Use the above method.
  std::vector<std::vector<CSCStripDigi> > stripDigis();
 
  /// returns one status digi per cfeb
  CSCCFEBStatusDigi statusDigi() const;

  unsigned short * data() {return theData;}
  unsigned sizeInWords() const {return theSize;} 
  unsigned boardNumber() const {return boardNumber_;}
  void setBoardNumber(int cfeb) {boardNumber_=cfeb;}
  
  friend std::ostream & operator<<(std::ostream & os, const CSCCFEBData &);
  static void selfTest();

  /// makes sure each time slice has a trailer
  bool check() const;
  
 private:
  unsigned short theData[1600];
  /// Shows where in theData the words start.  A bad slice will 
  /// be tagged with a false
  std::vector<std::pair<int,bool> > theSliceStarts;
  /// in words
  int theSize;
  unsigned boardNumber_;
  unsigned theNumberOfSamples;
  std::vector<uint16_t> bWords;
};

#endif
