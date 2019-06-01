#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader2005.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader2013.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>


CSCDMBHeader::CSCDMBHeader(uint16_t firmware_version)
:   theHeaderFormat(), theFirmwareVersion(firmware_version) 
{

  if (theFirmwareVersion == 2013) {
    theHeaderFormat = std::make_shared<CSCDMBHeader2013>();
  } else {
    theHeaderFormat = std::make_shared<CSCDMBHeader2005>();
  }

}

CSCDMBHeader::CSCDMBHeader(const uint16_t * buf, uint16_t firmware_version)
: theHeaderFormat(), theFirmwareVersion(firmware_version) 
{
  if (theFirmwareVersion == 2013) {
    theHeaderFormat = std::make_shared<CSCDMBHeader2013>(buf);
  } else {
    theHeaderFormat = std::make_shared<CSCDMBHeader2005>(buf);
  }
}

CSCDMBHeader2005 CSCDMBHeader::dmbHeader2005()   const {
  const CSCDMBHeader2005 * result = dynamic_cast<const CSCDMBHeader2005 *>(theHeaderFormat.get());
  if(result == nullptr)
  {
    throw cms::Exception("Could not get 2005 DMB header format");
  }
  return *result;
}


CSCDMBHeader2013 CSCDMBHeader::dmbHeader2013()   const {
  const CSCDMBHeader2013 * result = dynamic_cast<const CSCDMBHeader2013 *>(theHeaderFormat.get());
  if(result == nullptr)
  {
    throw cms::Exception("Could not get 2013 DMB header format");
  }
  return *result;
}



