#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData2006.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData2007.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cstring>  // for bzero

CSCAnodeData::CSCAnodeData(const CSCALCTHeader &header)  ///for digi->raw packing
    : firmwareVersion(header.alctFirmwareVersion()) {
  if (firmwareVersion == 2006) {
    theData = std::shared_ptr<CSCAnodeDataFormat>(new CSCAnodeData2006(header));
  } else {
    theData = std::shared_ptr<CSCAnodeDataFormat>(new CSCAnodeData2007(header));
  }
}

// initialize
CSCAnodeData::CSCAnodeData(const CSCALCTHeader &header, const unsigned short *buf)
    : firmwareVersion(header.alctFirmwareVersion()) {
  if (firmwareVersion == 2006) {
    theData = std::shared_ptr<CSCAnodeDataFormat>(new CSCAnodeData2006(header, buf));
  } else {
    theData = std::shared_ptr<CSCAnodeDataFormat>(new CSCAnodeData2007(header, buf));
  }
}

std::vector<std::vector<CSCWireDigi> > CSCAnodeData::wireDigis() const {
  std::vector<std::vector<CSCWireDigi> > result;
  for (int layer = 1; layer <= 6; ++layer) {
    result.push_back(wireDigis(layer));
  }
  return result;
}
