#ifndef EventFilter_CSCRawToDigi_CSCCFEBData_h
#define EventFilter_CSCRawToDigi_CSCCFEBData_h

class CSCCFEBTimeSlice;
class CSCStripDigi;
class CSCCFEBStatusDigi;

#include <vector>
#include <iosfwd>
#include <iostream>
#include <cstdint>

class CSCCFEBData {
public:
  /// read from an existing data stream.
  CSCCFEBData(unsigned boardNumber, const uint16_t *buf, uint16_t theFormatVersion = 2005, bool fDCFEB = false);
  /// create,
  CSCCFEBData(unsigned boardNumber, bool sixteenSamples, uint16_t theFormatVersion = 2005, bool fDCFEB = false);

  unsigned nTimeSamples() const { return theNumberOfSamples; }

  /// count from 0.  User should check if it's a bad slice
  const CSCCFEBTimeSlice *timeSlice(unsigned i) const;

  unsigned adcCounts(unsigned layer, unsigned channel, unsigned timeBin) const;
  unsigned adcOverflow(unsigned layer, unsigned channel, unsigned timeBin) const;
  unsigned controllerData(unsigned uglay, unsigned ugchan, unsigned timeBin) const;
  unsigned overlappedSampleFlag(unsigned layer, unsigned channel, unsigned timeBin) const;
  unsigned errorstat(unsigned layer, unsigned channel, unsigned timeBin) const;

  void add(const CSCStripDigi &, int layer);
  /// Fill strip digis for layer with raw detid = idlayer
  /// WARNING: these digis have no comparator information.

  ///faster way to get to digis
  void digis(uint32_t idlayer, std::vector<CSCStripDigi> &result) const;

  std::vector<CSCStripDigi> digis(unsigned idlayer) const;
  /// deprecated.  Use the above method.
  std::vector<std::vector<CSCStripDigi> > stripDigis();

  /// returns one status digi per cfeb
  CSCCFEBStatusDigi statusDigi() const;

  uint16_t *data() { return theData; }
  unsigned sizeInWords() const { return theSize; }
  unsigned boardNumber() const { return boardNumber_; }
  void setBoardNumber(int cfeb) { boardNumber_ = cfeb; }
  void setL1A(unsigned l1a);
  void setL1A(unsigned sample, unsigned l1a);

  friend std::ostream &operator<<(std::ostream &os, const CSCCFEBData &);
  static void selfTest();

  /// makes sure each time slice has a trailer
  bool check() const;

  bool isDCFEB() const { return fDCFEB; }

private:
  CSCCFEBTimeSlice *timeSlice(unsigned i);

  uint16_t theData[1600];
  /// Shows where in theData the words start.  A bad slice will
  /// be tagged with a false
  std::vector<std::pair<int, bool> > theSliceStarts;
  /// in words
  int theSize;
  unsigned boardNumber_;
  unsigned theNumberOfSamples;
  std::vector<uint16_t> bWords;
  uint16_t theFormatVersion;
  bool fDCFEB;
};

#endif
