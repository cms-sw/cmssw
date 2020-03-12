#ifndef CSCAnodeData_h
#define CSCAnodeData_h
#include <vector>
#include <memory>
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeDataFormat.h"

class CSCALCTHeader;

class CSCAnodeData {
public:
  /// a blank one, for Monte Carlo
  CSCAnodeData(const CSCALCTHeader &);
  /// fill from a real datastream
  CSCAnodeData(const CSCALCTHeader &, const unsigned short *buf);

  unsigned short *data() { return theData->data(); }
  /// the amount of the input binary buffer read, in 16-bit words
  unsigned short int sizeInWords() const { return theData->sizeInWords(); }

  /// input layer is from 1 to 6
  std::vector<CSCWireDigi> wireDigis(int layer) const { return theData->wireDigis(layer); }
  std::vector<std::vector<CSCWireDigi> > wireDigis() const;

  void add(const CSCWireDigi &wireDigi, int layer) { theData->add(wireDigi, layer); }

  static bool selfTest();

private:
  std::shared_ptr<CSCAnodeDataFormat> theData;
  int firmwareVersion;
};

#endif
