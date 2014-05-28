#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader2005.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader2013.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>


CSCDMBHeader::CSCDMBHeader(uint16_t firmware_version)
:   theHeaderFormat(), theFirmwareVersion(firmware_version) 
{

  if (theFirmwareVersion == 2013) {
    theHeaderFormat = boost::shared_ptr<CSCVDMBHeaderFormat>(new CSCDMBHeader2013());
  } else {
    theHeaderFormat = boost::shared_ptr<CSCVDMBHeaderFormat>(new CSCDMBHeader2005());
  }

}

CSCDMBHeader::CSCDMBHeader(unsigned short * buf, uint16_t firmware_version)
: theHeaderFormat(), theFirmwareVersion(firmware_version) 
{
  if (theFirmwareVersion == 2013) {
    theHeaderFormat = boost::shared_ptr<CSCVDMBHeaderFormat>(new CSCDMBHeader2013(buf));
  } else {
    theHeaderFormat = boost::shared_ptr<CSCVDMBHeaderFormat>(new CSCDMBHeader2005(buf));
  }
}

CSCDMBHeader2005 CSCDMBHeader::dmbHeader2005()   const {
  CSCDMBHeader2005 * result = dynamic_cast<CSCDMBHeader2005 *>(theHeaderFormat.get());
  if(result == 0)
  {
    throw cms::Exception("Could not get 2005 DMB header format");
  }
  return *result;
}


CSCDMBHeader2013 CSCDMBHeader::dmbHeader2013()   const {
  CSCDMBHeader2013 * result = dynamic_cast<CSCDMBHeader2013 *>(theHeaderFormat.get());
  if(result == 0)
  {
    throw cms::Exception("Could not get 2013 DMB header format");
  }
  return *result;
}



