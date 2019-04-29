#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer2005.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer2013.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>


CSCDMBTrailer::CSCDMBTrailer(uint16_t firmware_version)
:   theTrailerFormat(), theFirmwareVersion(firmware_version) 
{

  if (theFirmwareVersion == 2013) {
    theTrailerFormat = std::make_shared<CSCDMBTrailer2013>();
  } else {
    theTrailerFormat = std::make_shared<CSCDMBTrailer2005>();
  }

}

CSCDMBTrailer::CSCDMBTrailer(const uint16_t * buf, uint16_t firmware_version)
: theTrailerFormat(), theFirmwareVersion(firmware_version) 
{
  if (theFirmwareVersion == 2013) {
    theTrailerFormat = std::make_shared<CSCDMBTrailer2013>(buf);
  } else {
    theTrailerFormat = std::make_shared<CSCDMBTrailer2005>(buf);
  }
}

CSCDMBTrailer2005 CSCDMBTrailer::dmbTrailer2005()   const {
  const CSCDMBTrailer2005 * result = dynamic_cast<const CSCDMBTrailer2005 *>(theTrailerFormat.get());
  if(result == nullptr)
  {
    throw cms::Exception("Could not get 2005 DMB trailer format");
  }
  return *result;
}


CSCDMBTrailer2013 CSCDMBTrailer::dmbTrailer2013()   const {
  const CSCDMBTrailer2013 * result = dynamic_cast<const CSCDMBTrailer2013 *>(theTrailerFormat.get());
  if(result == nullptr)
  {
    throw cms::Exception("Could not get 2013 DMB trailer format");
  }
  return *result;
}



