#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer2005.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer2013.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>


CSCDMBTrailer::CSCDMBTrailer(uint16_t firmware_version)
:   theTrailerFormat(), theFirmwareVersion(firmware_version) 
{

  if (theFirmwareVersion == 2013) {
    theTrailerFormat = boost::shared_ptr<CSCVDMBTrailerFormat>(new CSCDMBTrailer2013());
  } else {
    theTrailerFormat = boost::shared_ptr<CSCVDMBTrailerFormat>(new CSCDMBTrailer2005());
  }

}

CSCDMBTrailer::CSCDMBTrailer(unsigned short * buf, uint16_t firmware_version)
: theTrailerFormat(), theFirmwareVersion(firmware_version) 
{
  if (theFirmwareVersion == 2013) {
    theTrailerFormat = boost::shared_ptr<CSCVDMBTrailerFormat>(new CSCDMBTrailer2013(buf));
  } else {
    theTrailerFormat = boost::shared_ptr<CSCVDMBTrailerFormat>(new CSCDMBTrailer2005(buf));
  }
}

CSCDMBTrailer2005 CSCDMBTrailer::dmbTrailer2005()   const {
  CSCDMBTrailer2005 * result = dynamic_cast<CSCDMBTrailer2005 *>(theTrailerFormat.get());
  if(result == 0)
  {
    throw cms::Exception("Could not get 2005 DMB trailer format");
  }
  return *result;
}


CSCDMBTrailer2013 CSCDMBTrailer::dmbTrailer2013()   const {
  CSCDMBTrailer2013 * result = dynamic_cast<CSCDMBTrailer2013 *>(theTrailerFormat.get());
  if(result == 0)
  {
    throw cms::Exception("Could not get 2013 DMB trailer format");
  }
  return *result;
}



